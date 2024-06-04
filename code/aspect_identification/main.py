import argparse
import wandb

from setup import check_gpu
from fine_tuning import fine_tuning
from prediction import prediction

if __name__ == "__main__":
    check_gpu()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run Fine-tuning")
    parser.add_argument("--predict", action="store_true", help="Run Prediction")
    parser.add_argument("--checkpoint", default="", type=str, help="The path of the model's checkpoint")
    args = parser.parse_args()
    
    wandb.init(project="Elden Ring 5_13", name="rbt-dapt-6k-aug-test")
    
    if args.train:
        fine_tuning(args)
    elif args.predict:
        prediction(args)
    else:
        raise ValueError("Fine-tuning or Prediction?")
