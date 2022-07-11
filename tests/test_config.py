

class NotInRange(Exception):
    def __init__(self, message="Value not in range"):
        self.message = message
        super().__init__(self.message)

def test_range():
    try:
        assert 1000 in range(1,2000)
    except:
        raise NotInRange