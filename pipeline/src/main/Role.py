#=====================Class Role, filters and scores job titles based on specified keywords=====================#

import re
class Role:
    def __init__(self, primary_title: str, min_score: int = 2, extra_titles: str = None) -> None:
        self.primary_title = primary_title
        self.current_score = 0
        self.min_score = min_score
        self.extra_titles = extra_titles
        if self.extra_titles is not None:
            self.keywords = self.primary_title.split(" ") + self.extra_titles.split(" ")
        else:
            self.keywords = self.primary_title.split(" ")

    def evaluate_role(self, job_title: str) -> int:
        search_patterns = [fr"(?=.*\b{term.lower()}\b)" for term in self.keywords]
        self.current_score = sum([bool(re.search(pattern, job_title)) for pattern in search_patterns])
        return self.current_score