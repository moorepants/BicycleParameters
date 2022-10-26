from bicycleparameters import io

def test_filename_to_dict():
    out = io.filename_to_dict('BrowserFrameCompoundFirst1.mat')
    ans = {'angleOrder': 'First',
           'bicycle': 'Browser',
           'part': 'Frame',
           'pendulum': 'Compound',
           'trial': '1'}
    assert out == ans
