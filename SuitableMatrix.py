import datetime

import numpy as np


class SuitableDatesMatrix:
    def __init__(self, total_weeks, total_time_slots):
        self.total_weeks = total_weeks
        self.total_time_slots = total_time_slots
        self.persona1_binary_matrix = np.zeros((total_weeks, total_time_slots), dtype=int)
        self.persona2_binary_matrix = np.zeros((total_weeks, total_time_slots), dtype=int)

    def update_matrix(self, week_number, persona1_suitable_dates, persona2_suitable_dates):
        if persona2_suitable_dates:

            week, day, start_str, end_str = persona1_suitable_dates
            start = datetime.datetime.strptime(start_str, "%I:%M%p").time()
            end = datetime.datetime.strptime(end_str, "%I:%M%p").time()
            start_slot = int(start.hour * 4 + start.minute / 15)
            end_slot = int(end.hour * 4 + end.minute / 15)
            self.persona1_binary_matrix[week_number - 1, start_slot:end_slot] = 1

            week, _, start_str, end_str = persona2_suitable_dates
            start = datetime.datetime.strptime(start_str, "%I:%M%p").time()
            end = datetime.datetime.strptime(end_str, "%I:%M%p").time()
            start_slot = int(start.hour * 4 + start.minute / 15)
            end_slot = int(end.hour * 4 + end.minute / 15)
            self.persona2_binary_matrix[week_number - 1, start_slot:end_slot] = 1

