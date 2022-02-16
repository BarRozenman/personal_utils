import sys
import time


def print_progress_bar(i, max, postText):
    n_bar = 30  # size of progress bar
    j = i / max
    if i == max:
        sys.stdout.write("\n")
    else:
        sys.stdout.write(
            f"\r[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}\r"
        )
        sys.stdout.flush()
