from django.shortcuts import render
from django.views import View


class Search(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'index.html')


class Results(View):
    def get(self, request, *args, **kwargs):
        query = request.GET.get('query')
        results = self.search_query(query)
        context = {"results": results, "query": query}
        return render(request, 'results.html', context=context)

    def search_query(self, query: str):
        # TODO: Implement query search by connecting to ML model
        data = \
            [
                {
                    "title": "title of some document",
                    "line_matches": ["line1", "line2", "line3"],
                    "similarity_score": 0.9,
                },
                {
                    "title": "title of other document",
                    "line_matches": ["line1", "line2", "line3"],
                    "similarity_score": 0.8,
                },
                {
                    "title": "title of another document",
                    "line_matches": ["line1", "line2", "line3"],
                    "similarity_score": 0.95,
                },
            ]

        return data
