"""
CLI entrypoint for running the MO-FJSP optimizer.
Examples:
    python src/main.py --mode rl --generations 10 --pop-size 10
    PYTHONPATH=src python -m main --mode nsga --generations 10 --pop-size 10
"""
from ga_optimizer import main as optimizer_main


if __name__ == "__main__":
    optimizer_main()
