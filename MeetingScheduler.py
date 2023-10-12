import numpy as np

class CommonAvailableTimesMatrix:
    def __init__(self, total_weeks, total_time_slots):
        self.total_weeks = total_weeks
        self.total_time_slots = total_time_slots
        self.common_binary_matrix = np.zeros((total_weeks, total_time_slots), dtype=int)


    def update_common_matrix(self, week_number, common_available_times):
        for week, day, start, end in common_available_times:
            start_slot = int(start.hour * 4 + start.minute / 15)
            end_slot = int(end.hour * 4 + end.minute / 15)
            self.common_binary_matrix[week_number - 1, start_slot:end_slot] = 1
