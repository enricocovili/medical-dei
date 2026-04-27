import time
import pathlib
# import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="PaddleOCR Inference")
    # parser.add_argument("input_path", type=str, help="Path to input image or folder")
    # args = parser.parse_args()

    input_path = "imgs/output/postprocess/images/"


    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    start = time.time()
    import paddleocr

    pipeline = paddleocr.PaddleOCRVL()
    end = time.time()
    print(f"Loading model time: {end - start:.2f} seconds")

    if pathlib.Path(input_path).is_file():
        print(f"Processing image: {input_path}", end="... ")
        start = time.time()
        output = pipeline.predict(input_path)
        end = time.time()
        print(f"Inference time: {end - start:.2f} seconds")
        for res in output:
            res.print()
            res.save_to_json(
                save_path=f"imgs/output/paddle-ocr/{pathlib.Path(input_path).stem}.json"
            )
            res.save_to_img(
                save_path=f"imgs/output/paddle-ocr/{pathlib.Path(input_path).stem}.png"
            )

    else:
        pathdir = pathlib.Path(input_path)
        for img_path in pathlib.Path(pathdir).glob("*"):
            output = pipeline.predict(str(img_path))
            end = time.time()
            print(f"Inference time: {end - start:.2f} seconds")
            for res in output:
                res.print()
                res.save_to_json(
                    save_path=f"imgs/output/paddle-ocr/{img_path.stem}.json"
                )
                res.save_to_img(save_path=f"imgs/output/paddle-ocr/{img_path.stem}.png")
