## Django Search App

### Creating conda env for django app:
1. conda create -n django_search python=3.8
2. conda install -n django_search django=3.2
3. conda activate django_search

### Cloning this app and start django server
1. git clone https://github.com/Shubham-Gaikwad23/doc_search.git
2. cd doc_search\docSearch
3. python manage.py collectstatic
4. python manage.py runserver
5. Open  http://127.0.0.1:8000/ in browser
