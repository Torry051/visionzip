import runpy
from llava.model import builder
from visionzip import visionzip

def run(module="llava.eval.model_vqa_loader"):
    original_loader = builder.load_pretrained_model

    def load_compressed_model(*args, **kwargs):
        tokenizer, model, processor, context_len = original_loader(*args, **kwargs)
        model = visionzip(model, dominant=54, contextual=10)
        print("VisionZip enabled: 54 dominant + 10 contextual tokens", flush=True)

        def show_token_count(module, inputs, output):
            print("Projected visual tokens:", output.shape[-2], flush=True)

        model.get_model().mm_projector.register_forward_hook(show_token_count)
        return tokenizer, model, processor, context_len

    builder.load_pretrained_model = load_compressed_model
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        builder.load_pretrained_model = original_loader


if __name__ == "__main__":
    run()
