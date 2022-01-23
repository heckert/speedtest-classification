import src.utils
import src.dirs

def test_get_filenames_in_dir():
    nb_dir = src.dirs.project_dir / 'notebooks'
    files = src.utils.get_filenames_in_dir(nb_dir)
    assert len(files) > 0

