from pathlib import Path
import requests
from loguru import logger
from tqdm import tqdm
import typer

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw",
):
    """
    Downloads the two restaurant review datasets from the public GCS bucket
    into the raw data directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://storage.googleapis.com/remla-group-5-unique-bucket"
    files = [
        "a1_RestaurantReviews_HistoricDump.tsv",
        "a2_RestaurantReviews_FreshDump.tsv",
    ]

    logger.info(f"Downloading datasets to {output_dir.resolve()}")

    for filename in tqdm(files, desc="Downloading files"):
        url = f"{base_url}/{filename}"
        dest_path = output_dir / filename

        try:
            logger.info(f"Downloading {filename}...")
            reponse = requests.get(url, stream=True, timeout=30)
            with open(dest_path, "wb") as f:
                for chunk in reponse.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.success(f"Saved to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

    logger.success("All downloads completed.")


if __name__ == "__main__":
    app()
