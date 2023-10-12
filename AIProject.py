import datetime
import random
import numpy as np

from Project.AIMainProject.MeetingPredictor import MeetingScheduler
from Project.AIMainProject.SuitableMatrix import SuitableDatesMatrix

np.set_printoptions(threshold=np.inf)

# Print the entire binary matrix

from Project.AIMainProject.MeetingScheduler import CommonAvailableTimesMatrix


class Persona:
    def __init__(self, name="", work_schedule=None, preferred_days=[], avoids=[], days_off=[],
                 meeting_during_work=False):
        self.name = name
        if work_schedule is None:
            work_schedule = ("8:00AM", "5:00PM")  # Default work_schedule
        self.work_schedule = self.validate_and_fix_work_schedule(work_schedule)
        preferred_days = [self.get_day_number(day) for day in preferred_days]
        avoids = [self.get_day_number(day) for day in avoids]
        days_off = [self.get_day_number(day) for day in days_off]
        self.preferred_days = list(set(preferred_days) - set(avoids))
        self.avoids = list(set(avoids) - set(preferred_days))
        self.days_off = days_off
        self.meetings_during_work = meeting_during_work
        # Initialize a dictionary to store weekly calendars for the year
        self.weekly_calendars = {}
        self.personal_commitments = {}

    def validate_and_fix_work_schedule(self, work_schedule):
        if isinstance(work_schedule, tuple) and len(work_schedule) == 2:
            start_time, end_time = work_schedule
            if isinstance(start_time, str) and isinstance(end_time, str):
                try:
                    start_time = datetime.datetime.strptime(start_time, "%I:%M%p").time()
                    end_time = datetime.datetime.strptime(end_time, "%I:%M%p").time()
                    if start_time < end_time:
                        return work_schedule
                except ValueError:
                    pass  # Invalid time format, fall through to the default below
        # If any validation fails, return the default schedule
        return ("8:00AM", "5:00PM")

    def update_work_schedule(self, start_time, end_time):
        self.work_schedule = (start_time, end_time)

    def get_day_number(self, day):
        dictionary = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7
        }
        return dictionary.get(day)

    def generate_weekly_calendar(self, week_number, freedays_and_times):
        # Update the weekly calendar for the specified week
        if week_number not in self.weekly_calendars:
            self.weekly_calendars[week_number] = {}

        for day, times in freedays_and_times.items():
            day_number = self.get_day_number(day)
            if day_number is not None:
                self.weekly_calendars[week_number][day_number] = times

    def generate_probability_matrix(self, start_week, end_week):
        year_probability_matrix = []

        for week_number in range(start_week, end_week + 1):
            # Get the weekly calendar for the specified week
            weekly_calendar = self.weekly_calendars.get(week_number, {})

            # Initialize an empty probability matrix for the week
            probability_matrix = []

            # Define the time range from 4:00 AM to 10:00 PM in 15-minute intervals
            start_time = datetime.datetime.strptime("4:00AM", "%I:%M%p").time()
            end_time = datetime.datetime.strptime("10:00PM", "%I:%M%p").time()
            delta = datetime.timedelta(minutes=15)

            for day in range(1, 8):  # Loop through days (1 for Monday, 2 for Tuesday, etc.)
                day_probs = []
                current_time_dt = datetime.datetime.combine(datetime.date(1, 1, 1),
                                                            start_time)  # Convert to datetime object

                while current_time_dt.time() <= end_time:
                    # Initialize probability as 0.2
                    probability = 0.2

                    # Check if the day corresponds to a preferred day
                    if day in self.preferred_days:
                        probability += 0.5  # Add a higher probability if it's a preferred day

                    # Check if the day corresponds to an avoided day
                    if day in self.avoids:
                        probability = 0.0  # Set probability to 0 if it's an avoided day

                    # Check if the day corresponds to a day off
                    if day in self.days_off:
                        probability = 0.0  # Set probability to 0 if it's a day off

                    # Get the current time as a datetime.time object
                    current_time = current_time_dt.time()

                    # Check if the current time falls within the work schedule
                    if self.work_schedule:
                        work_start, work_end = self.work_schedule
                        work_start_time = datetime.datetime.strptime(work_start, "%I:%M%p").time()
                        work_end_time = datetime.datetime.strptime(work_end, "%I:%M%p").time()

                        if work_start_time <= current_time <= work_end_time:
                            # If the time is within the work schedule and meetings are during work, set probability to 1
                            if self.meetings_during_work:
                                probability = 1.0
                            else:
                                # If meetings are not during work, set probability to 0
                                probability = 0.0

                            # Check if it's a free day_and_time
                            if day in weekly_calendar:
                                for start, end in weekly_calendar[day]:
                                    start_time_dt = datetime.datetime.strptime(start, "%I:%M%p").time()
                                    end_time_dt = datetime.datetime.strptime(end, "%I:%M%p").time()

                                    # Calculate the time difference in minutes
                                    time_difference = (end_time_dt.hour * 60 + end_time_dt.minute) - \
                                                      (start_time_dt.hour * 60 + start_time_dt.minute)

                                    if time_difference <= 180 and day not in self.days_off:
                                        # If the free time is <= 3 hours, make it 1 even if it falls under work schedule
                                        probability = 1.0
                                    else:
                                        if day not in self.days_off:
                                            # Otherwise, make it 0.5
                                            probability = 0.4
                                        else:
                                            probability = 0

                    # Check if the current time is before 4:00 AM or after 10:00 PM
                    if current_time < datetime.time(4, 0) or current_time > datetime.time(22, 0):
                        probability *= 0.2  # Reduce probability for times outside the 4 AM to 10 PM range

                    day_probs.append(probability)

                    # Add delta to current_time_dt
                    current_time_dt += delta

                probability_matrix.append(day_probs)

            year_probability_matrix.append(probability_matrix)

        return year_probability_matrix

    @staticmethod
    def time_range(start, end, delta):
        current_time = start
        while current_time <= end:
            yield current_time
            current_time += delta

    @staticmethod
    def is_time_within_range(current_time, time_ranges):
        for start, end in time_ranges:
            if start <= current_time <= end:
                return True
        return False

    @staticmethod
    def random_time(start, end):
        # Generate random hours (in 12-hour clock format) and minutes
        hours = random.randint(start.hour, end.hour)
        minutes = random.randint(0, 3) * 15  # Choose 0, 15, 30, or 45 minutes

        # Determine AM or PM based on the hour
        am_pm = "AM" if hours < 12 else "PM"

        # Adjust hours for PM if needed
        if am_pm == "PM" and hours > 12:
            hours -= 12

        # Format the time as a string with leading zeros if necessary
        formatted_hours = f"{hours:02d}"
        formatted_minutes = f"{minutes:02d}"

        # Combine hours, minutes, and AM/PM into a time string
        time_str = f"{formatted_hours}:{formatted_minutes}{am_pm}"

        return time_str

    @staticmethod
    def random_days():
        all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        num_preferred = random.randint(0, len(all_days))
        preferred_days = random.sample(all_days, num_preferred)

        num_avoids = random.randint(0, len(all_days))
        avoids = random.sample(all_days, num_avoids)

        return preferred_days, avoids

    # Function to generate a random list of preferred and avoided days

    # Generate n personas with random attributes
    def generate_random_personas(self, n):
        personas = []
        for i in range(n):
            name = f"Person_{i + 1}"

            # Generate random start and end times for the work schedule
            work_start = self.random_time(datetime.time(4, 0), datetime.time(12, 0))
            work_end = self.random_time(datetime.time(13, 0), datetime.time(20, 0))

            # Generate random preferred days and avoids
            preferred_days, avoids = self.random_days()

            # Generate a random week number (e.g., from 1 to 52)
            week_number = random.randint(1, 52)

            # Create a Persona object with the generated attributes
            persona = Persona(name, (work_start, work_end), preferred_days, avoids)

            # Generate a random weekly calendar for the persona and assign it to the random week
            freedays_and_times = self.generate_random_freedays_and_times()  # You can customize this dictionary as needed
            persona.generate_weekly_calendar(week_number, freedays_and_times)

            # Append the persona to the list
            personas.append(persona)

        return personas

    def generate_random_freedays_and_times(self):
        freedays_and_times = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Randomly choose a subset of days for which the persona has free time
        num_free_days = random.randint(0, len(days))
        free_days = random.sample(days, num_free_days)

        # For each free day, generate a random start and end time in the correct format
        for day in free_days:
            # Generate random hours (in 12-hour clock format) and minutes
            start_hours = random.randint(1, 12)
            start_minutes = random.randint(0, 3) * 15  # 0, 15, 30, or 45 minutes
            end_hours = random.randint(1, 12)
            end_minutes = random.randint(0, 3) * 15

            # Generate random AM or PM
            start_am_pm = random.choice(["AM", "PM"])
            end_am_pm = random.choice(["AM", "PM"])

            # Format the time strings correctly
            start_time = f"{start_hours:02d}:{start_minutes:02d}{start_am_pm}"
            end_time = f"{end_hours:02d}:{end_minutes:02d}{end_am_pm}"

            # Check if the start time comes after the end time, and swap them if needed
            start_datetime = datetime.datetime.strptime(start_time, "%I:%M%p")
            end_datetime = datetime.datetime.strptime(end_time, "%I:%M%p")
            if start_datetime >= end_datetime:
                start_time, end_time = end_time, start_time

            freedays_and_times[day] = [(start_time, end_time)]

        return freedays_and_times

    def simulate_realistic_person_calendar(self, name, num_weeks):
        personas = []
        initial_work_schedule = self.generate_realistic_work_schedule()

        # Initialize a list to store week-numbered probability matrices
        weekly_matrices = []

        for week_number in range(1, num_weeks + 1):
            # Randomly decide whether to use the initial schedule or introduce a variation
            if random.random() < 0.4:  # 20% chance of variation
                work_schedule = self.generate_realistic_work_schedule()
            else:
                work_schedule = initial_work_schedule

            # Generate random preferred days and avoids for this week
            preferred_days, avoids = self.random_days()

            # Randomly add personal commitments for this week
            self.add_random_personal_commitments(week_number)

            # Randomly set days off, with a preference for Sundays and Saturdays
            days_off = self.random_days_off()

            meeting_during_work = random.choice([True, False])
            # Create a Persona object for this week with the generated attributes
            persona = Persona(name, work_schedule, preferred_days, avoids, days_off=days_off, meeting_during_work=True)

            # Generate a random weekly calendar for this week's persona
            freedays_and_times = self.generate_random_freedays_and_times()

            # Apply personal commitments for this week if they exist
            if week_number in self.personal_commitments:
                for commitment in self.personal_commitments[week_number]:
                    day, start_time, end_time = commitment
                    freedays_and_times[day] = [(start_time, end_time)]

            persona.generate_weekly_calendar(week_number, freedays_and_times)

            # Append the persona to the list
            personas.append(persona)

            # Generate the probability matrix for this week
            week_probability_matrix = persona.generate_probability_matrix(week_number, week_number)

            # Append the week number and its probability matrix as a tuple
            weekly_matrices.append((week_number, week_probability_matrix))

        return persona, weekly_matrices

    def add_random_personal_commitments(self, week_number):
        # Randomly add personal commitments for the week
        personal_commitments = []
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Randomly choose the number of commitments for the week (between 0 and 3)
        num_commitments = random.randint(0, 3)

        for _ in range(num_commitments):
            day = random.choice(days)
            start_time = self.random_time(datetime.time(4, 0), datetime.time(20, 0))
            end_time = self.random_time(datetime.time(4, 0), datetime.time(20, 0))

            # Ensure that start_time is before end_time
            if start_time > end_time:
                start_time, end_time = end_time, start_time

            personal_commitments.append((day, start_time, end_time))

        # Store the personal commitments for the week
        self.personal_commitments[week_number] = personal_commitments

    def random_days_off(self):
        # Randomly set days off, with a preference for Sundays and Saturdays
        days_off = []

        # Randomly choose whether to have days off
        if random.random() < 0.7:  # 70% chance of having days off
            # Add either Sunday, Saturday, or both to days off
            if random.random() < 0.5:  # 50% chance of Sunday
                days_off.append("Sunday")
            if random.random() < 0.5:  # 50% chance of Saturday
                days_off.append("Saturday")
        else:
            # No days off or other random days (30% chance)
            other_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            num_other_days_off = random.randint(0, len(other_days))
            days_off.extend(random.sample(other_days, num_other_days_off))

        return days_off

    def generate_realistic_work_schedule(self):
        # Generate a more realistic work schedule with longer durations
        work_start = self.random_time(datetime.time(8, 0), datetime.time(10, 0))
        work_end = self.random_time(datetime.time(16, 0), datetime.time(18, 0))
        return (work_start, work_end)

    def find_common_available_times(self, other_persona, start_week, end_week):

        common_available_times = []

        for week_num in range(start_week, end_week + 1):
            # Generate random freedays_and_times for both personas for this week
            self_freedays_and_times = self.generate_random_freedays_and_times()
            other_freedays_and_times = other_persona.generate_random_freedays_and_times()

            # Generate probability matrices for both personas for this week with the updated freedays_and_times
            self.generate_weekly_calendar(week_num, self_freedays_and_times)
            other_persona.generate_weekly_calendar(week_num, other_freedays_and_times)
            self_probability_matrix = self.generate_probability_matrix(week_num, week_num)
            other_probability_matrix = other_persona.generate_probability_matrix(week_num, week_num)

            for day_num in range(7):  # Loop through days (1 for Monday, 2 for Tuesday, etc.)
                current_start_time = None
                current_end_time = None

                for time_slot in range(len(self_probability_matrix[0][day_num])):
                    # Check if both personas are available at this time slot
                    self_prob = self_probability_matrix[0][day_num][time_slot]
                    other_prob = other_probability_matrix[0][day_num][time_slot]

                    if self_prob >= 0.5 and other_prob >= 0.5:
                        # Convert time_slot to minutes
                        minutes = time_slot * 15

                        # Create datetime objects for start and end times
                        start_datetime = datetime.datetime(2023, 1, 1, 4, 0) + datetime.timedelta(minutes=minutes)
                        end_datetime = start_datetime + datetime.timedelta(minutes=15)

                        # Extract time components
                        start_time = start_datetime.time()
                        end_time = end_datetime.time()

                        # Check if this slot continues the current range
                        if start_time == current_end_time:
                            current_end_time = end_time
                        else:
                            # If not, append the current range and start a new one
                            if current_start_time is not None:
                                common_available_times.append(
                                    (week_num, self.get_day_name(day_num), current_start_time, current_end_time))
                            current_start_time = start_time
                            current_end_time = end_time

                # Append the last continuous time range if any
                if current_start_time is not None:
                    common_available_times.append(
                        (week_num, self.get_day_name(day_num), current_start_time, current_end_time))

        return common_available_times

    def select_suitable_date(self, common_available_times):
        # Define priorities for different criteria (can vary for each persona)
        priorities = {
            "Work Hours": 4,
            "Preferred Days": 3,
            "Avoided Days": -2,  # Penalize avoided days
            "Personal Commitments": -3,  # Penalize time slots with personal commitments
            # Add more priorities specific to the persona
        }

        # Create a dictionary to store scores for each available time slot
        scores = {}

        # Loop through each available time slot and calculate a score based on criteria
        for week, day, start, end in common_available_times:
            score = 0

            # Apply priorities for this persona's criteria
            if self.is_within_work_hours(start, end):
                score += priorities["Work Hours"]
            if self.get_day_number(day) in self.preferred_days:
                score += priorities["Preferred Days"]
            if self.get_day_number(day) in self.avoids:
                score += priorities["Avoided Days"]

            # Check if the time slot conflicts with personal commitments (e.g., doctor's appointments)
            if self.has_personal_commitment(week, day, start, end):
                score += priorities["Personal Commitments"]

            # Store the score for this time slot
            scores[(week, day, start, end)] = score

        # Sort available times by their scores in descending order
        sorted_times = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select the time slot with the highest score as the suitable date
        if sorted_times:
            week, day, start, end = sorted_times[0][0]
            return week, day, start.strftime('%I:%M%p'), end.strftime('%I:%M%p')
            # return f"Week {week}, {day}: {start.strftime('%I:%M%p')} - {end.strftime('%I:%M%p')}"
        else:
            return ()
            return "No suitable date found."

    def add_personal_commitment(self, week_number, day, start_time, end_time):
        # Add a personal commitment for a specific week, day, and time slot
        if week_number not in self.personal_commitments:
            self.personal_commitments[week_number] = []

        self.personal_commitments[week_number].append((day, start_time, end_time))

    def has_personal_commitment(self, week_number, day, start_time, end_time):
        # Check if there is a personal commitment for a specific week, day, and time slot
        if week_number in self.personal_commitments:
            commitment = (day, start_time, end_time)
            if commitment in self.personal_commitments[week_number]:
                return True
        return False

    def get_day_name(self, dayNum):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return days[dayNum]

    def is_within_work_hours(self, start_time, end_time):
        # Ensure start_time and end_time are datetime.time objects
        if not isinstance(start_time, datetime.time) or not isinstance(end_time, datetime.time):
            raise ValueError("Both start_time and end_time should be datetime.time objects.")

        # Convert work schedule times to datetime.time objects
        work_start_time = datetime.datetime.strptime(self.work_schedule[0], "%I:%M%p").time()
        work_end_time = datetime.datetime.strptime(self.work_schedule[1], "%I:%M%p").time()

        # Check if the time slot falls within the persona's work hours
        return start_time >= work_start_time and end_time <= work_end_time


# Create two Persona instances with default values


persona = Persona()
persona2 = Persona()

weekstart = 0
weekend = 1000
# Call the simulate_realistic_person_calendar method to generate personas
persona, weekly_matrices = persona.simulate_realistic_person_calendar("John", weekend - weekstart)
persona2, weekly_matrices2 = persona2.simulate_realistic_person_calendar("Akwesi", weekend - weekstart)

# Initialize a list to store the common available times for each week
common_available_times_list = []

# Iterate through the weeks and calculate common available times

for week_number in range(weekstart, weekend):
    common_available_times = persona.find_common_available_times(persona2, week_number, week_number)
    common_available_times_list.append(common_available_times)

# Example usage:
total_weeks = weekend - weekstart  # Adjust based on your data
total_time_slots = 24 * 4  # 15-minute intervals for a day

# Create an instance of the CommonAvailableTimesMatrix class
common_times_matrix = CommonAvailableTimesMatrix(total_weeks, total_time_slots)

total_weeks = weekend - weekstart
total_time_slots = 7 * 96
suitable_dates_matrix = SuitableDatesMatrix(total_weeks, total_time_slots)
# Iterate through common available times and update the binary matrix
for week_number, common_available_times in enumerate(common_available_times_list, start=1):
    common_times_matrix.update_common_matrix(week_number, common_available_times)
    persona1_suitable_dates = persona.select_suitable_date(common_available_times)
    persona2_suitable_dates = persona2.select_suitable_date(common_available_times)
    suitable_dates_matrix.update_matrix(week_number, persona1_suitable_dates, persona2_suitable_dates)

persona1_binary_matrix = suitable_dates_matrix.persona1_binary_matrix
persona2_binary_matrix = suitable_dates_matrix.persona2_binary_matrix

predictor = MeetingScheduler()
accuracy_persona1, accuracy_persona2 = predictor.predict_suitable_dates(
    common_times_matrix.common_binary_matrix, persona1_binary_matrix, persona2_binary_matrix)

print(accuracy_persona1)
print(accuracy_persona2)