import gradio as gr
import subprocess
import os
import glob
import shutil

# --- Configuration ---
OUTPUT_DIR = "./outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_inference(
    input_audio,
    # General Options
    output_format,
    selected_stems,
    # Hardware & Audio
    large_gpu,
    input_gain,
    restore_gain,
    filter_vocals,
    # Inference Parameters
    big_shifts,
    overlap_demucs,
    overlap_vocft,
    overlap_insthq4,
    # Models to Use
    use_bs, weight_bs,
    use_mel, weight_mel,
    use_instvoc, weight_instvoc,
    use_vit, weight_vit,
    use_vocft, weight_vocft,
    use_insthq4, weight_insthq4
):
    if input_audio is None:
        raise gr.Error("⚠️ Please upload an audio file first!")

    # 1. Clean previous outputs
    print("Cleaning output directory...")
    for f in glob.glob(f"{OUTPUT_DIR}/*"):
        try: os.remove(f)
        except: pass

    # 2. Determine "Vocals Only" mode
    # If the user ONLY wants Vocals, we can speed things up by passing --vocals_only
    # If they want Drums/Bass/Other, we must NOT pass it.
    wants_backing_stems = any(stem in ["Drums", "Bass", "Other"] for stem in selected_stems)
    vocals_only_flag = not wants_backing_stems
    
    # 3. Build the Command
    cmd = [
        "python", "inference.py",
        "--input_audio", input_audio,
        "--output_folder", OUTPUT_DIR,
        "--output_format", output_format,
        "--BigShifts", str(big_shifts),
        "--overlap_demucs", str(overlap_demucs),
        "--overlap_VOCFT", str(overlap_vocft),
        "--overlap_InstHQ4", str(overlap_insthq4),
        "--input_gain", str(input_gain),
    ]

    # Flags
    if large_gpu: cmd.append("--large_gpu")
    if restore_gain: cmd.append("--restore_gain")
    if filter_vocals: cmd.append("--filter_vocals")
    if vocals_only_flag: cmd.append("--vocals_only")

    # Dynamic Model Injection
    # We only append flags if the user checked the "Use" box
    if use_bs:
        cmd.append("--use_BSRoformer")
        cmd.extend(["--weight_BSRoformer", str(weight_bs)])
        # Note: You can expose BSRoformer_model selection if you have multiple checkpoints
        cmd.extend(["--BSRoformer_model", "ep_317_1297"]) 

    if use_mel:
        cmd.append("--use_Kim_MelRoformer")
        cmd.extend(["--weight_Kim_MelRoformer", str(weight_mel)])

    if use_instvoc:
        cmd.append("--use_InstVoc")
        cmd.extend(["--weight_InstVoc", str(weight_instvoc)])

    if use_vit:
        cmd.append("--use_VitLarge")
        cmd.extend(["--weight_VitLarge", str(weight_vit)])

    if use_vocft:
        cmd.append("--use_VOCFT")
        cmd.extend(["--weight_VOCFT", str(weight_vocft)])

    if use_insthq4:
        cmd.append("--use_InstHQ4")
        cmd.extend(["--weight_InstHQ4", str(weight_insthq4)])

    print(f"🚀 Executing Command:\n{' '.join(cmd)}")
    
    # 4. Run Process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Real-time output capturing could be done with yield, but for simplicity we wait
    stdout, stderr = process.communicate()
    
    logs = f"✅ Execution Complete.\n\nSTDOUT:\n{stdout}"
    if stderr:
        logs += f"\n\n⚠️ STDERR:\n{stderr}"

    # 5. Retrieve Files
    # Helper to find file by keyword (vocal, drum, bass, etc)
    def find_file(keyword):
        # backend produces names like "filename_vocals.flac"
        files = glob.glob(f"{OUTPUT_DIR}/*_{keyword}.{output_format.lower()}")
        return files[0] if files else None

    # We map the results to the players
    # "instrum" is the backend name for Instrumental
    return (
        logs,
        gr.update(value=find_file("vocals"), visible=("Vocals" in selected_stems)),
        gr.update(value=find_file("instrum"), visible=("Instrumental" in selected_stems)),
        gr.update(value=find_file("drums"), visible=("Drums" in selected_stems)),
        gr.update(value=find_file("bass"), visible=("Bass" in selected_stems)),
        gr.update(value=find_file("other"), visible=("Other" in selected_stems)),
    )


# --- Theme & CSS ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_shadow="*shadow_drop_lg",
    block_title_text_weight="600"
)

css = """
h1 { text-align: center; color: #3b82f6; }
.contain { max-width: 1200px; margin: auto; }
"""

# --- UI Construction ---
with gr.Blocks(title="MVSEP-MDX23 Studio", theme=theme, css=css) as app:
    
    gr.Markdown("# 🎚️ MVSEP-MDX23 Studio")
    gr.Markdown("### Professional Music Source Separation | Ensemble Logic")

    with gr.Row():
        
        # === LEFT COLUMN: CONTROLS ===
        with gr.Column(scale=5):
            
            # --- Input ---
            with gr.Group():
                input_audio = gr.Audio(label="Input Audio", type="filepath", sources=["upload"])
                
                with gr.Row():
                    selected_stems = gr.CheckboxGroup(
                        label="Target Stems",
                        choices=["Vocals", "Instrumental", "Drums", "Bass", "Other"],
                        value=["Vocals", "Instrumental"],
                        interactive=True
                    )
                    output_format = gr.Dropdown(
                        label="Format", 
                        choices=["FLAC", "PCM_16", "FLOAT"], 
                        value="FLAC", 
                        interactive=True
                    )
            
            # --- TABS ---
            with gr.Tabs():
                
                # TAB 1: MODEL BLENDER (The Weights)
                with gr.TabItem("🎛️ Model Mixer"):
                    gr.Markdown("Mix your AI models. **Blending Power (0-100)** determines how much influence a model has.")
                    gr.Markdown("_Note: If you enable 2 models at 100 power, they are mixed 50/50._")
                    
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            use_bs = gr.Checkbox(label="Use BSRoformer", value=True)
                            weight_bs = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=80)
                        
                        with gr.Column(variant="panel"):
                            use_mel = gr.Checkbox(label="Use Kim MelRoformer", value=True)
                            weight_mel = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=100)
                    
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            use_instvoc = gr.Checkbox(label="Use InstVoc", value=True)
                            weight_instvoc = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=30)

                        with gr.Column(variant="panel"):
                            use_vit = gr.Checkbox(label="Use VitLarge", value=False)
                            weight_vit = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=10)

                    with gr.Row():
                        with gr.Column(variant="panel"):
                            use_vocft = gr.Checkbox(label="Use VOC-FT", value=False)
                            weight_vocft = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=10)

                        with gr.Column(variant="panel"):
                            use_insthq4 = gr.Checkbox(label="Use InstHQ4", value=False)
                            weight_insthq4 = gr.Slider(label="Blending Power", minimum=0, maximum=100, value=10)


                # TAB 2: ADVANCED ENGINEERING
                with gr.TabItem("🚀 Engineering"):
                    
                    gr.Markdown("### ⚙️ Inference Quality")
                    with gr.Row():
                        big_shifts = gr.Slider(
                            label="BigShifts", minimum=1, maximum=11, step=1, value=3, 
                            info="Repeats processing with time-shifts to reduce artifacts. (Higher = Slower)"
                        )
                        filter_vocals = gr.Checkbox(
                            label="Filter Vocals (<50Hz)", value=False,
                            info="High-pass filter to clean muddy low-end from vocals."
                        )

                    gr.Markdown("### 🎚️ Gain Staging")
                    with gr.Row():
                        input_gain = gr.Slider(label="Input Gain (dB)", minimum=-20, maximum=20, value=0)
                        restore_gain = gr.Checkbox(label="Restore Gain", value=True, info="Apply negative gain after processing to match original volume.")

                    gr.Markdown("### 💻 Hardware")
                    large_gpu = gr.Checkbox(
                        label="Large GPU Mode", value=True, 
                        info="Keeps all models in VRAM. Disable if you have <11GB VRAM."
                    )
                    
                    with gr.Accordion("Detailed Overlap Settings (Context Window)", open=False):
                        with gr.Row():
                            overlap_demucs = gr.Slider(label="Demucs Overlap", minimum=0.0, maximum=0.99, value=0.1)
                            overlap_vocft = gr.Slider(label="VOC-FT Overlap", minimum=0.0, maximum=0.99, value=0.1)
                            overlap_insthq4 = gr.Slider(label="InstHQ4 Overlap", minimum=0.0, maximum=0.99, value=0.1)

            # --- Action ---
            btn = gr.Button("⚡ Start Separation Process", variant="primary", size="lg")


        # === RIGHT COLUMN: OUTPUTS ===
        with gr.Column(scale=4):
            gr.Markdown("### 📻 Results")
            
            out_voc = gr.Audio(label="Vocals", type="filepath", visible=False)
            out_inst = gr.Audio(label="Instrumental", type="filepath", visible=False)
            out_drum = gr.Audio(label="Drums", type="filepath", visible=False)
            out_bass = gr.Audio(label="Bass", type="filepath", visible=False)
            out_other = gr.Audio(label="Other", type="filepath", visible=False)

            with gr.Accordion("Process Logs", open=True):
                logs = gr.Textbox(show_label=False, lines=15, interactive=False, text_align="left")

    # --- Bindings ---
    btn.click(
        fn=run_inference,
        inputs=[
            input_audio, output_format, selected_stems,
            large_gpu, input_gain, restore_gain, filter_vocals,
            big_shifts, overlap_demucs, overlap_vocft, overlap_insthq4,
            use_bs, weight_bs,
            use_mel, weight_mel,
            use_instvoc, weight_instvoc,
            use_vit, weight_vit,
            use_vocft, weight_vocft,
            use_insthq4, weight_insthq4
        ],
        outputs=[logs, out_voc, out_inst, out_drum, out_bass, out_other]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)