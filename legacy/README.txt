These are the original stage-by-stage scripts, kept only for reference.
Everything they do is now in the package, reachable from main.py:

    rtsp_view.py      ->  python main.py --mode view
    detect_person.py  ->  python main.py --mode detect
    track_person.py   ->  python main.py --mode track

They import each other by module name, so they only run from inside this
folder. Safe to delete this whole directory once you are happy with main.py.
