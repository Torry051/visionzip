"""Use the installed LLaVA loader in isolated baseline/compressed processes."""
import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["baseline", "visionzip64"], required=True)
    mode_args, loader_args = parser.parse_known_args()
    from llava.model import builder
    original_loader = builder.load_pretrained_model

    def load_model(*args, **kwargs):
        tokenizer, model, processor, context_len = original_loader(*args, **kwargs)
        if mode_args.mode == "visionzip64":
            from visionzip import visionzip
            model = visionzip(model, dominant=54, contextual=10)
            print("VisionZip enabled: 54 dominant + 10 contextual tokens", flush=True)
        expected = 64 if mode_args.mode == "visionzip64" else 576

        def check_tokens(module, inputs, output):
            count = int(output.shape[-2])
            if count != expected:
                raise RuntimeError(f"Expected {expected} visual tokens, observed {count}")
            print("Projected visual tokens:", count, flush=True)

        model.get_model().mm_projector.register_forward_hook(check_tokens)
        return tokenizer, model, processor, context_len

    builder.load_pretrained_model = load_model
    sys.argv = ["llava.eval.model_vqa_loader", *loader_args]
    runpy.run_module("llava.eval.model_vqa_loader", run_name="__main__")


if __name__ == "__main__":
    main()
