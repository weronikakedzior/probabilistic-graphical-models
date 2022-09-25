from scipy.stats import t as tstudent
import torch
from IPython.display import Code, display
import inspect


def make_ols(x_train, y_train, x_test, alpha=0.1):
    n_train = x_train.shape[0]
    p = x_train.shape[1]
    columns = x_train.columns
    x_train_intercept = torch.tensor(x_train.to_numpy())
    y_train = torch.tensor(y_train.to_numpy())
    x_test_intercept = torch.tensor(x_test.to_numpy())

    gramian_matrix = torch.matmul(x_train_intercept.t(), x_train_intercept)
    inv_xx = torch.inverse(gramian_matrix)

    beta = torch.matmul(
        torch.matmul(inv_xx, x_train_intercept.t()),
        y_train
    )

    y_hat_train = torch.matmul(beta, x_train_intercept.t())
    y_hat_test = torch.matmul(beta, x_test_intercept.t())

    sigma = torch.sqrt(
        torch.sum((y_train - y_hat_train) ** 2) / (y_train.shape[0] - p)
    ).item()

    q_bottom = tstudent(n_train - p).ppf(alpha / 2)
    q_up = tstudent(n_train - p).ppf(1 - alpha / 2)

    ci_train = []
    ci_test = []
    pi_train = []
    pi_test = []

    for idx, row in enumerate(x_train_intercept):
        var_mean = torch.matmul(
            torch.matmul(
                row,
                inv_xx
            ),
            row.t()
        )
        ci_train.append((
            y_hat_train[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean).item(),
            y_hat_train[idx].item() + q_up * sigma * torch.sqrt(var_mean).item()
        ))
        pi_train.append((
            y_hat_train[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean + 1).item(),
            y_hat_train[idx].item() + q_up * sigma * torch.sqrt(
                var_mean + 1).item()
        ))

    for idx, row in enumerate(x_test_intercept):
        var_mean = torch.matmul(
            torch.matmul(
                row,
                inv_xx
            ),
            row.t()
        )
        ci_test.append((
            y_hat_test[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean).item(),
            y_hat_test[idx].item() + q_up * sigma * torch.sqrt(var_mean).item()
        ))
        pi_test.append((
            y_hat_test[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean + 1).item(),
            y_hat_test[idx].item() + q_up * sigma * torch.sqrt(
                var_mean + 1).item()
        ))

    return (
        (y_hat_train, torch.tensor(ci_train), torch.tensor(pi_train)),
        (y_hat_test, torch.tensor(ci_test), torch.tensor(pi_test)),
        dict(zip(columns, beta.numpy())),
        sigma,
    )


def display_sourcecode(fun):
    display(Code(inspect.getsource(fun), language='python3'))
