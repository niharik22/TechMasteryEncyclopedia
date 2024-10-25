from datetime import datetime


class JobLocationParser:
    def __init__(self):
        self.us_states = ['IA', 'KS', 'UT', 'VA', 'NC', 'NE', 'SD', 'AL', 'ID', 'FM', 'DE', 'AK', 'CT', 'PR', 'NM',
                          'MS', 'PW', 'CO', 'NJ', 'FL', 'MN', 'VI', 'NV', 'AZ', 'WI', 'ND', 'PA', 'OK', 'KY',
                          'RI', 'NH', 'MO', 'ME', 'VT', 'GA', 'GU', 'AS', 'NY', 'CA', 'HI', 'IL', 'TN', 'MA', 'OH',
                          'MD', 'MI', 'WY', 'WA', 'OR', 'MH', 'SC', 'IN', 'LA', 'MP', 'DC', 'MT', 'AR', 'WV', 'TX']
        self.can_province_abbrev = {
            'Alberta': 'AB',
            'British Columbia': 'BC',
            'Manitoba': 'MB',
            'New Brunswick': 'NB',
            'Newfoundland and Labrador': 'NL',
            'Northwest Territories': 'NT',
            'Nova Scotia': 'NS',
            'Nunavut': 'NU',
            'Ontario': 'ON',
            'Prince Edward Island': 'PE',
            'Quebec': 'QC',
            'Saskatchewan': 'SK',
            'Yukon': 'YT',
            'Labrador': 'NL',  # Labrador part of Newfoundland and Labrador
            'Newfoundland': 'NL'  # Newfoundland part of Newfoundland and Labrador
        }
        self.countries = ['USA', 'Canada']

    def extract_place_of_work(self, location):
        """Helper function to extract place of work (Hybrid/On-site/Remote) if present in the location string."""
        work_types = ['Hybrid', 'On-site', 'Remote']
        for work_type in work_types:
            if f"({work_type})" in location:
                location = location.replace(f"({work_type})", "").strip()  # Remove place of work from location
                return work_type, location
        return "NA", location

    def is_state_or_province(self, location_part):
        """Helper function to check if a given part of the location is a state or province."""
        if location_part in self.us_states:
            return location_part, 'USA'
        for province, abbrev in self.can_province_abbrev.items():
            if location_part == province or location_part == abbrev:
                return abbrev, 'Canada'
        return None, None

    def categorize_location(self, location):
        # Step 1: Extract place of work (Hybrid/On-site/Remote) and clean location
        place_of_work, location = self.extract_place_of_work(location)

        # Step 2: Split by comma to extract city, state/region, and potentially country
        location_map = location.split(",")
        city, state, country = "All", "All", "All"  # Default values

        if len(location_map) >= 2:  # Handle cases with city and state/province
            city = location_map[0].strip()  # Assume city is the first part
            state_info = location_map[1].split()  # Second part could be state/province

            # Check if the second part is a valid state or province
            potential_state = state_info[0].strip()
            state_from_second_part, inferred_country = self.is_state_or_province(potential_state)

            if state_from_second_part:
                # Case: We found a valid state in the second part
                state = state_from_second_part
                country = inferred_country
            else:
                # Check if there is a third part for country information
                if len(location_map) >= 3:
                    potential_country = location_map[2].strip()
                    if potential_country in self.countries:
                        country = potential_country

        elif len(location_map) == 1:  # Only one part present (maybe just state or province)
            potential_state = location_map[0].strip()
            state_from_first_part, inferred_country = self.is_state_or_province(potential_state)

            if state_from_first_part:
                # Case: Single part is a valid state or province, no city
                state = state_from_first_part
                country = inferred_country
            else:
                # Case: No valid state or province
                city = location_map[0].strip()  # Keep it as city
                state = "All"

        return {
            "city": city,
            "state": state,
            "country": country,
            "place_of_work": place_of_work
        }

    def process_job_locations(self, job_dict, config):

        # Get today's date and time
        scraped_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for url, job_data in job_dict.items():
            # Get the location from the job data
            location = job_data['location']

            # Categorize the location into city, state, country, and place_of_work
            categorized_location = self.categorize_location(location)

            # Add new fields to the job data
            job_data['city'] = categorized_location['city']
            job_data['state'] = categorized_location['state']
            # job_data['country'] = categorized_location['country']
            job_data['country'] = config["scraping"]["search"]["country"]  # country from config
            job_data['place_of_work'] = categorized_location['place_of_work']

            # Add the scraped_date field with today's date
            job_data['scraped_date'] = scraped_date

        return job_dict
