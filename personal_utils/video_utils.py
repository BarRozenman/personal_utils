import skvideo.io


def subsample_and_write(filename, out_filename, n_steps):
    video_mat = skvideo.io.vread(filename)  # returns a NumPy array
    video_mat = video_mat[::n_steps]  # subsample
    skvideo.io.vwrite(out_filename, video_mat)


def split_video_and_write(filename, out_filename, start, end):
    video_mat = skvideo.io.vread(filename)  # returns a NumPy array
    video_mat = video_mat[..., start:end]  # subsample
    skvideo.io.vwrite(out_filename, video_mat)


if __name__ == '__main__':
        pass
