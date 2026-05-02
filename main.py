from dataclasses import dataclass, field

import numpy as np
from lectrace import text

from _utils import Datapoint, load_electricity_dataset


@dataclass
class Parameter:
    W: np.ndarray = field(default_factory=list)
    b: float = 0


_DATASET_PATH = "electricity.csv"  # @hide


def main():
    text("# Gradient Descent Explained: Math and Code, Side by Side")
    intro()
    explain_data()
    # exaples
    eg = get_dataset(5)
    explain_loaded()
    explain_design()
    param = Parameter()  # @inspect
    # param.W = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # @inspect
    param.W = np.random.randn(eg[0].X.shape[0])  # using random weights @inspect
    # param.W = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # @inspect
    param.b = 1  # @inspect
    steps = 10  # @inspect
    learning_rate = 0.01  # @inspect
    explain_loss()
    explain_gradient()
    explain_update()
    for i in range(1, steps + 1):  # @inspect
        loss = calc_avg_loss(eg, param)  # @inspect loss @stepover
        grad = calc_avg_grad_loss(eg, param)  # @inspect grad @stepover
        param = Parameter(  # @inspect
            W=param.W - learning_rate * grad[0],
            b=param.b - learning_rate * grad[1],
        )
        text(f"""
        **Step {i} of {steps}** - watch the variables panel on the right.
        loss is coming down, W and b are shifting with every step.
        That is gradient descent working.
        """)
    closing()


def intro():
    text("## Introduction")
    text("""
    ### Who am I?
    - Author: **PraiseGod D. Adesanmi**
    - Github: https://github.com/praisegee
    - LinkedIn: https://linkedin.com/in/praisegod
    """)
    text("### Why I Wrote This")
    text("""
    When I started learning machine learning, the math was the hardest part for me.
    Not because I could not read the formulas, but because I could not see what they
    were actually doing. You look at something like

    $\\frac{\\partial L}{\\partial W} = 2r \\cdot X$

    and it just sits there. A partial derivative. Of L. With respect to W. Okay.
    But what does that mean in practice? What does it look like in code?
    """)
    text(
        "That gap between the formula and the implementation was what kept tripping me up."
    )
    text("""
    The calculus part was the worst. I understood the chain rule in theory from school
    but applying it to a loss function with vectors involved felt different. When you
    have a scalar loss depending on a vector of weights, suddenly you are not taking
    one derivative, you are taking a derivative with respect to each weight separately.
    That took me a while to really sit with.
    """)
    text("""
    The vector operations were the other thing. The dot product X . W looked simple
    enough on paper but in code it is X @ W, and I kept second guessing what shape
    things were, what was getting multiplied by what, whether the dimensions lined up.
    I had to slow down and trace through it manually before it made sense.
    """)
    text("""
    The moment it clicked for me was when I realised that dW = 2r * X is just saying:
    the gradient points in the same direction as X, scaled by how wrong our prediction
    was. If r is large, we were very wrong and we take a big step. If r is small,
    we were close and we take a small step. Once I saw that, the code wrote itself.
    """)
    text("""
    That is what pushed me to explain it this way. I wanted to show the formula and
    the code that implements it side by side, step by step, so the connection is
    impossible to miss.
    """)


def explain_data():
    text("""
    Before we get into the math, lets understand the data we are working with.
    Everything in machine learning starts with data, so it helps to know
    where it comes from and what it actually looks like.
    """)
    text("""
    ## Where the Data Comes From

    The dataset is from Kaggle. It tracks electricity demand across two states
    in Australia, New South Wales and Victoria, recorded every 30 minutes.
    """)
    text("""
    You can find it here:
    https://www.kaggle.com/datasets/ulrikthygepedersen/electricity-demands
    """)
    text("""
    When you open the CSV file, the first row is the header. It lists the column names:

        date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, class

    That is 9 columns total. The first 8 will be our input features X.
    The last one, class, is what we want to predict. It says either UP or DOWN,
    meaning electricity demand went up or down compared to the previous period.
    """)
    text("""
    ## The Raw CSV Looks a Bit Odd

    If you peek inside the file, an actual data row looks like this:

        0.0, b'2', 0.0, 0.056443, 0.439155, 0.003467, 0.422915, 0.414912, b'UP'
    """)
    text("""
    Notice the b'' around some values. That is a Python bytes notation. It means
    the value was stored as a byte string instead of a regular string.
    b'2' is just the number 2. b'UP' is just the text UP. Nothing fancy.
    """)
    text("""
    The _process_data function in _utils.py cleans this up before we use the data.
    For every value in a row, it does two things.
    """)
    text("""
    First, it strips the b'' wrapper if it is there:

        if isinstance(val, str) and "b" in val.lower():
            val = val[2:-1]   # b'UP'  becomes  UP
                              # b'2'   becomes  2
    """)
    text("""
    Second, it converts the cleaned value to a float.
    Most values convert just fine. The ones that dont are UP and DOWN,
    since those are words not numbers. So those get encoded:

        try:
            val = float(val)
        except ValueError:
            val = 1 if "UP" in val.upper() else 0

    UP becomes 1.0. DOWN becomes 0.0. Now every value in the row is a number.
    """)


def explain_loaded():
    text("""
    ## What We Got Back

    After loading, we have 5 Datapoint objects. Think of each one as a single row
    from the CSV, split into two parts:

        @dataclass
        class Datapoint:
            X: np.ndarray  # the 8 input features as a numpy array
            y: float       # the label, either 0.0 (DOWN) or 1.0 (UP)
    """)
    text("""
    So for the first row in our sample it would look like:

        Datapoint(
            X = [0.0, 2.0, 0.0, 0.056, 0.439, 0.003, 0.422, 0.414],
            y = 1.0   # demand went UP in this period
        )
    """)
    text("""
    X is a numpy array with 8 values, one per feature column.
    y is a single float, the thing our model will try to predict.
    """)
    text("Now we have data. Next step is to set up the model.")


def explain_design():
    text("""
    ## Why Model It as a Class

    You might wonder why we bother making a Datapoint class instead of just
    keeping X and y as two separate numpy arrays.
    """)
    text("""
    The reason is readability. When you write a function like calc_loss, you
    want it to be obvious what is coming in. Compare these two:

        # without the class
        def calc_loss(X, y, W, b):
            residual = X @ W + b - y

        # with the class
        def calc_loss(d: Datapoint, p: Parameter):
            residual = d.X @ p.W + p.b - d.y
    """)
    text("""
    The second one reads almost like the math formula on paper.
    d.X is the features for that datapoint. d.y is its label.
    You do not have to remember which argument is which or what index holds what.
    """)
    text("""
    A dataclass is the right tool here because we are not adding any behaviour,
    we are just grouping related values under a meaningful name. The `@dataclass`
    decorator handles the boilerplate so we can write:

        Datapoint(X=some_array, y=1.0)

    instead of writing an __init__ method ourselves.
    """)
    text("""
    Think of it like a named container. X and y always belong together because
    they describe the same sample. Putting them in a class makes that explicit.
    """)
    text("""
    ## The Model

    The goal is to find parameters W and b so that the linear model

    $\\hat{y} = X \\cdot W + b$

    gives predictions close to the true values y in our dataset.
    """)
    text("""
    W is a weight vector with one value per feature. It decides how much each
    feature contributes to the prediction. b is a bias term that shifts the
    result up or down.
    """)
    text(
        "We do not solve for W and b directly. Instead we use gradient descent, start with a rough guess and improve step by step."
    )
    text("""
    ## Why Parameter is Also a Class

    Same thinking applies to Parameter. W and b always travel together.
    Every function that makes a prediction needs both of them, every function
    that computes a gradient receives both of them, and every update step
    changes both of them at once.
    """)
    text("""
    So instead of passing them around as two separate arguments:

        def calc_loss(d, W, b):
            ...

    we bundle them:

        def calc_loss(d: Datapoint, p: Parameter):
            residual = d.X @ p.W + p.b - d.y
    """)
    text("p.W and p.b. Clean, and it mirrors how you would write it in the math.")
    text("""
    Another thing: at the end of each training step we create a brand new
    Parameter object with the updated values:

        param = Parameter(
            W = param.W - learning_rate * grad[0],
            b = param.b - learning_rate * grad[1],
        )

    We do not mutate the old one in place. This makes it easier to reason
    about what changed between steps, and if you ever want to keep a history
    of parameters over time, you already have each one as its own object.
    """)


def explain_loss():
    text("""
    ## Loss Function

    The loss for a single sample measures how far off our prediction is.
    We square the difference so it is always positive and so bigger errors
    get penalized more:

    $L^{(i)} = \\left( X^{(i)} \\cdot W + b - y^{(i)} \\right)^2$
    """)
    text("""
    This is what calc_loss computes:

        residual = X . W + b - y
        loss     = residual ** 2
    """)
    text("""
    Averaging over all m samples gives us the cost J. This is the single
    number we are trying to bring down:

    $J = \\frac{1}{m} \\sum_{i=1}^{m} L^{(i)}$
    """)


def explain_gradient():
    text("""
    ## Gradient

    To reduce J we need to know which direction to nudge W and b.
    We do this by computing how J changes with respect to each parameter.
    """)
    text("""
    For one sample, let r be the residual: r = X . W + b - y.
    By the chain rule the partial derivatives of L are:

    $\\frac{\\partial L}{\\partial W} = 2r \\cdot X$

    $\\frac{\\partial L}{\\partial b} = 2r$
    """)
    text("""
    This is exactly what calc_grad_loss computes:

        r  = X . W + b - y
        dr = 2 * r
        dW = dr * X    # derivative w.r.t W
        db = dr * 1    # derivative w.r.t b
    """)
    text("""
    We then average dW and db over all m samples:

    $dW = \\frac{1}{m} \\sum_{i=1}^{m} \\frac{\\partial L^{(i)}}{\\partial W}, \\quad
    db = \\frac{1}{m} \\sum_{i=1}^{m} \\frac{\\partial L^{(i)}}{\\partial b}$
    """)


def explain_update():
    text("""
    ## Parameter Update

    Now we move W and b in the direction that reduces J:

    $W \\leftarrow W - \\alpha \\cdot dW$

    $b \\leftarrow b - \\alpha \\cdot db$
    """)
    text("""
    Alpha is the learning rate. It controls how big each step is.
    Too large and we overshoot. Too small and training crawls.
    We use 0.01 here.

    In code:

        W = W - learning_rate * dW
        b = b - learning_rate * db
    """)


def closing():
    text("""
    ## What We Covered

    We started from raw CSV data, cleaned it up, and built a linear model that
    learns from it using gradient descent. Every piece of it was written from
    scratch in plain Python and numpy, no libraries doing the heavy lifting.
    """)
    text("""
    The three core ideas to take away:

    - The model predicts with $\\hat{y} = X \\cdot W + b$
    - The loss tells us how wrong we are: $J = \\frac{1}{m} \\sum L^{(i)}$
    - The gradient tells us which way to move W and b to fix that
    """)
    text("""
    Each training step applies those three ideas in sequence, and after 10
    iterations you can already see the loss coming down.
    """)
    text("""
    ## What is Coming Next

    This is linear regression. It works by fitting a straight line to the data.
    But our target here is actually a binary label, UP or DOWN, which means a
    straight line is not really the right tool for the job.
    """)
    text("""
    Next up is logistic regression. Instead of predicting a raw number, we pass
    the output through a sigmoid function that squashes it to a value between 0
    and 1. That gives us a proper probability we can threshold into a class label.
    """)
    text("""
    After that we will start stacking layers. One linear layer followed by a
    non-linear activation, then another layer on top. That is a neural network.
    The gradient descent stays the same but the chain rule gets longer, and that
    longer chain is what people call backpropagation.
    """)
    text("""
    So the plan is:

    1. Logistic regression (proper binary classification)
    2. A single hidden layer network (the simplest neural net)
    3. Backpropagation written by hand so you can see every step
    """)
    text("""
    Each one will be presented the same way as this one. Code you can step
    through line by line, with the math shown right next to the piece of code
    that implements it.
    """)
    text("""
    ## About This Presentation

    This walkthrough was built with [lectrace](https://github.com/praisegee/lectrace),
    an open source Python library developed by the same author of this lecture,
    `lectrace` turns regular Python scripts into interactive step-through presentations.
    """)
    text("""
    You write your code normally, add a few text() calls where you want
    explanations to appear, and lectrace traces the execution and builds a
    viewer where you can step forward and backward through every line,
    watching variables change in real time alongside the explanation.
    """)
    text(
        "The whole thing runs in the browser. No slides, no screenshots, just the actual code running."
    )
    text("""
    ## Try It Yourself

    The best way to understand what is happening here is to run it yourself
    and change things. Clone the source, open main.py, and start experimenting:

    - Change the initial weights in param.W from all ones to all zeros
    - Set learning_rate to 0.1 and watch the loss shoot around
    - Set learning_rate to 0.0001 and watch it barely move
    - Increase steps from 10 to 100 and see how far the loss comes down
    - Load more data by changing get_dataset(5) to get_dataset(50)
    """)
    text("""
    Each change will show up in the variable panel as you step through.
    That is the whole point of the viewer. Nothing is hidden.
    """)
    text("""
    Source code:
    https://github.com/praisegee/gradient-descent
    """)
    text("""
    ## Sources

    Interactive viewer:
    https://praisegee.github.io/gradient-descent
    """)
    text("""
    lectrace on GitHub:
    https://github.com/praisegee/lectrace
    """)


def get_dataset(n: int = 5):  # @inspect
    text(
        f"We call `load_electricity_dataset` with the path to our CSV and `head = {n}`."
    )
    text(f"""
    The head parameter limits how many rows we load. The full dataset has
    thousands of rows, but we only need a small sample to understand the
    structure and test our code. {n} rows is enough for that. Once the logic
    is correct, you can remove the limit and train on everything.
    """)
    text(f"""
    Inside that function, `_read_csv opens` the file and reads every row into a list:

        data = list(csv.reader(f))
        # data[0] is the header row, the column names
        # data[1 : head + 1] gives us the first {n} rows of actual data
    """)
    text("""
    Each row comes back as a plain list of strings. CSV files are just text,
    so everything is a string at this point. Thats why we need _process_data
    to convert each value to a float before we can do any math.
    """)
    return load_electricity_dataset(_DATASET_PATH, head=n)


def calc_loss(d: Datapoint, p: Parameter):  # @inspect
    residual = (d.X @ p.W + p.b) - d.y
    loss = residual**2
    return loss


def calc_avg_loss(datapoints: list[Datapoint], param: Parameter) -> np.ndarray:
    avg_losses = [calc_loss(datapoint, param) for datapoint in datapoints]
    return np.mean(avg_losses)


def calc_grad_loss(d: Datapoint, p: Parameter):  # @inspect
    r = (d.X @ p.W + p.b) - d.y
    dr = 2 * r
    dW = dr * d.X
    db = dr * 1
    return dW, db


def calc_avg_grad_loss(datapoints: list[Datapoint], param: Parameter):
    gradients = [calc_grad_loss(datapoint, param) for datapoint in datapoints]
    dW_vector = [gradient[0] for gradient in gradients]
    db_vector = [gradient[1] for gradient in gradients]
    avg_dW = np.mean(dW_vector, axis=0)
    avg_db = np.mean(db_vector)
    return avg_dW, avg_db


if __name__ == "__main__":
    main()
