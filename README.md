# Gradient Descent Explained: Math and Code, Side by Side

By [PraiseGod D. Adesanmi](https://github.com/praisegee) · [LinkedIn](https://linkedin.com/in/praisegod)

A step-by-step walkthrough of linear regression trained with gradient descent,
built on real electricity demand data.

## What it covers

- Loading and cleaning a CSV dataset
- The linear model and what W and b actually mean
- The loss function and why we square the error
- Computing gradients using the chain rule
- Updating parameters one step at a time

## View the lecture

Interactive viewer: https://praisegee.github.io/gradient-descent

## How to run locally

Install dependencies:

```bash
uv sync
```

Build and serve the interactive trace:

```bash
lectrace build && lectrace serve
```

Then open the link in your browser. Use the arrow keys to step through the code
line by line. The variables panel on the right updates as you go.

## Dataset

Electricity demand in New South Wales and Victoria, Australia.
Source: https://www.kaggle.com/datasets/ulrikthygepedersen/electricity-demands

## Built with

[lectrace](https://github.com/praisegee/lectrace) - an open source Python library
by PraiseGod D. Adesanmi that turns Python scripts into interactive code walkthroughs
you can step through in the browser.
