import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    evaluator = torch.load("saved_models/evaluator.pth")
    output_root = "plots"

    result = evaluator.evaluate()
    result_df = result.as_dataframe()
    with open(f"{output_root}/result.csv", "w") as f:
        f.write(result_df.to_csv())

    result.visualize()
    plt.savefig(f"{output_root}/results.png")
    plt.close("all")

    print(result_df)
