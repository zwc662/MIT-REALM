from clfrl.utils.schedules import Constant, JoinSched, join_sched


def test_join_sched():
    c1, c2 = 3.7, 5.8
    sched2_start = 8
    sched1 = Constant(c1)
    sched2 = Constant(c2)
    sched = JoinSched(sched1, sched2, sched2_start).make()

    for ii in range(20):
        if ii < sched2_start:
            assert sched(ii) == c1
        else:
            assert sched(ii) == c2


if __name__ == "__main__":
    test_join_sched()
