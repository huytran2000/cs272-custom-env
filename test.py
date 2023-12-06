# import pygame


class Animal:
    leg_count = 1  # got overwritten

    def __init__(self, leg_count) -> None:
        self.leg_count = leg_count
        self.l = [1, 2, 3, 4]
        # sself.count

    def printt(self):
        self.l = [[1, 1, 1], [2, 2]]
        print(self.l[1][0])
        # self.l = a

    def leg_count1(self):
        return self.leg_count**2
