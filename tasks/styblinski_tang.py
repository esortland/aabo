import sys 
sys.path.append("../")
import torch
from tasks.objective import Objective
from botorch.test_functions.synthetic import StyblinskiTang
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyblinskiTangObjective(Objective):
    """Styblinsky function optimization task
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.neg_st = StyblinskiTang(negate=True)

        super().__init__(
            dim=40,
            lb=-5,
            ub=5,
            **kwargs,
        )

    def f(self, x):
        x = x.to(device)
        self.num_calls += 1
        y = self.neg_st(x)
        return y.item()


if __name__ == "__main__":
    obj = StyblinskiTangObjective()
    known_optimum = torch.tensor(obj.neg_st.optimal_value).to(dtype=obj.dtype)
    known_optimum = known_optimum.unsqueeze(0)
    best_objective_value = obj(known_optimum)
    print(f"Best possible StyblinskiTang val: {best_objective_value}")


