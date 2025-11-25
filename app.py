import voxcpm
import os
import numpy as np
import torch
import gradio as gr
import spaces
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM-0.5B"


class VoxCPMDemo:
    def __init__(self) -> None:
        # Check if user wants to force CPU mode
        force_cpu = os.environ.get(
            "VOXCPM_FORCE_CPU", "").lower() in ("1", "true", "yes")

        if force_cpu:
            print("🔧 CPU mode forced by VOXCPM_FORCE_CPU environment variable")
            self.device = "cpu"
        else:
            # Test if CUDA actually works (not just if it's available)
            self.device = self._get_working_device()

        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        try:
            self.asr_model: Optional[AutoModel] = AutoModel(
                model=self.asr_model_id,
                disable_update=True,
                log_level='DEBUG',
                device="cuda:0" if self.device == "cuda" else "cpu",
            )
        except Exception as e:
            print(
                f"⚠️  Warning: Failed to initialize ASR model on {self.device}: {e}")
            print("   Falling back to CPU for ASR model...")
            self.device = "cpu"
            self.asr_model: Optional[AutoModel] = AutoModel(
                model=self.asr_model_id,
                disable_update=True,
                log_level='DEBUG',
                device="cpu",
            )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM-0.5B"

    def _get_working_device(self) -> str:
        """Test if CUDA actually works, fallback to CPU if not."""
        if not torch.cuda.is_available():
            return "cpu"

        # Test if CUDA actually works by trying a simple operation
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda"
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            if "kernel image" in error_str.lower() or "CUDA" in error_str:
                print(f"⚠️  CUDA compatibility issue detected: {error_str}")
                print(
                    "   Your PyTorch was compiled for a different GPU compute capability.")
                print("   Automatically falling back to CPU mode...")
                print("   (Set VOXCPM_FORCE_CPU=1 to skip this check in the future)")
                # Hide CUDA devices to prevent models from trying to use them
                if "CUDA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                print(f"⚠️  CUDA error: {e}")
                print("   Falling back to CPU mode...")
            return "cpu"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                    os.makedirs(target_dir, exist_ok=True)
                    print(
                        f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                    snapshot_download(
                        repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(
                        f"Warning: HF download failed: {e}. Falling back to 'data'.")
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")

        # Try to load model, catch CUDA/Triton compatibility errors and fallback to CPU
        try:
            self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir)
            print("Model loaded successfully.")
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            # Check for CUDA/Triton compatibility errors (including BackendCompilerFailed)
            is_cuda_error = (
                "CUDA" in error_str or
                "cuda" in error_str.lower() or
                "kernel image" in error_str.lower() or
                "triton" in error_str.lower() or
                "compute capability" in error_str.lower() or
                ("capability" in error_str.lower() and ("6." in error_str or "too old" in error_str.lower())) or
                "backendcompilerfailed" in str(type(e)).lower() or
                "inductor" in error_str.lower()
            )

            if is_cuda_error:
                print(f"\n❌ CUDA/Triton compatibility error detected:")
                print(f"   {error_str}")
                print("\n🔄 Automatically falling back to CPU mode...")

                # Force CPU mode
                self.device = "cpu"
                os.environ["VOXCPM_FORCE_CPU"] = "1"

                # Hide CUDA devices to prevent further CUDA attempts
                # Always set this when falling back to CPU due to compatibility issues
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

                # Suppress torch dynamo errors as suggested in the error message
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                except (ImportError, AttributeError):
                    pass

                print(
                    "   Retrying model initialization with CPU mode and optimization disabled...")

                # Retry loading with CPU mode and optimization disabled
                try:
                    self.voxcpm_model = voxcpm.VoxCPM(
                        voxcpm_model_path=model_dir,
                        optimize=False  # Disable torch.compile which uses Triton
                    )
                    print("✅ Model loaded successfully on CPU mode.")
                except Exception as retry_error:
                    print(f"\n❌ Failed to load model on CPU: {retry_error}")
                    raise RuntimeError(
                        "Failed to load model even on CPU mode. "
                        "This might indicate a more serious issue with the model files or dependencies."
                    ) from retry_error
            else:
                # Re-raise if it's not a CUDA/Triton error
                raise

        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        try:
            res = self.asr_model.generate(
                input=prompt_wav, language="auto", use_itn=True)
            text = res[0]["text"].split('|>')[-1]
            return text
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            if "CUDA" in error_str or "cuda" in error_str.lower() or "kernel image" in error_str.lower():
                print(f"⚠️  ASR model CUDA error: {error_str}")
                print(
                    "   ASR recognition failed. You can manually enter the prompt text.")
                return ""  # Return empty string so user can manually enter text
            else:
                # Re-raise if it's not a CUDA error
                raise

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (16000, wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo."""
    # static assets (logo path) - optional, may not be available in all Gradio versions
    try:
        gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])
    except (AttributeError, TypeError):
        pass  # Feature not available in this Gradio version

    # CSS styles - will be added via HTML component for compatibility
    # (We don't use theme parameter to ensure compatibility with all Gradio versions)
    css_styles = """
    <style>
    .logo-container {
        text-align: center;
        margin: 0.5rem 0 1rem 0;
    }
    .logo-container img {
        height: 80px;
        width: auto;
        max-width: 200px;
        display: inline-block;
    }
    /* Bold accordion labels */
    #acc_quick details > summary,
    #acc_tips details > summary {
        font-weight: 600 !important;
        font-size: 1.1em !important;
    }
    /* Bold labels for specific checkboxes */
    #chk_denoise label,
    #chk_denoise span,
    #chk_normalize label,
    #chk_normalize span {
        font-weight: 600;
    }
    </style>
    """

    # Create Blocks with no optional parameters for maximum compatibility
    # We'll add styling via HTML/CSS inside the interface instead
    with gr.Blocks() as interface:
        # Add CSS via HTML component (works in all Gradio versions)
        gr.HTML(css_styles)
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide ｜快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use ｜使用说明
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.
               **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).
               **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
            3. **Enter target text** - Type the text you want the model to speak.
               **输入目标文本** - 输入您希望模型朗读的文字内容。
            4. **Generate Speech** - Click the "Generate" button to create your audio.
               **生成语音** - 点击"生成"按钮，即可为您创造出音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips ｜使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Prompt Speech Enhancement｜参考语音降噪
            - **Enable** to remove background noise for a clean, studio-like voice, with an external ZipEnhancer component.
              **启用**：通过 ZipEnhancer 组件消除背景噪音，获得更好的音质。
            - **Disable** to preserve the original audio's background atmosphere.
              **禁用**：保留原始音频的背景环境声，如果想复刻相应声学环境。

            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.
              **启用**：使用 WeTextProcessing 组件，可处理常见文本。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input ({HH AH0 L OW1}), try it!
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如 {da4}{jia1}好）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive.
              **调低**：如果提示语音听起来不自然或过于夸张。
            - **Higher CFG** for better adherence to the prompt speech style or input text.
              **调高**：为更好地贴合提示音频的风格或输入文本。

            ### Inference Timesteps｜推理时间步
            - **Lower** for faster synthesis speed.
              **调低**：合成速度更快。
            - **Higher** for better synthesis quality.
              **调高**：合成质量更佳。
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="Prompt Speech (Optional, or let VoxCPM improvise)",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                    info="We use ZipEnhancer model to denoise the prompt audio."
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                        label="Target Text",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="Text Normalization",
                        elem_id="chk_normalize",
                        info="We use wetext library to normalize the input text."
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value,
                    inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=demo.prompt_wav_recognition, inputs=[
                          prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "localhost", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10).launch(server_name=server_name,
                                        server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    server_port = int(os.environ.get("SERVER_PORT", "7860"))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
    run_demo(server_name=server_name, server_port=server_port)
