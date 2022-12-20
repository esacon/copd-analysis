import os 

src_path = './Audio Samples'
dst_path = './Audio Samples/samples'

folders = ['luis', 'juan', 'jorgerico', 'jorgerigu', 'jacque']

def get_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_file_path(filename):
    return os.path.join(get_path(), filename)

def move_files():
    for folder in folders:
        for filename in os.listdir(os.path.join(src_path, folder)):
            src = os.path.join(src_path, folder, filename)
            dst = os.path.join(dst_path, filename)
            os.rename(src, dst)


if __name__ == '__main__':
    move_files()