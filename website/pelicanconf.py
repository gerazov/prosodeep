#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'bgerazov'
SITENAME = 'ProsoDeep'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DISPLAY_PAGES_ON_MENU = False
DISPLAY_CATEGORIES_ON_MENU = False
STATIC_PATHS = ['images']
TYPOGRIFY = True

# for a no blog website
TAGS_SAVE_AS = ''
TAG_SAVE_AS = ''

# GitHub  - fix this?
GITHUB_URL = 'https://github.com/bgerazov/prosodeep'
GITHUB_POSITION = 'right'

# class PAGE:
#     def __init__(self, title, url):
#         self.title = title
#         self.url = url

PAGES = (
    ('Home', ''),
    ('Project', 'project'),
    ('PySFC', 'pysfc'),
    ('WSFC', 'wsfc'),
    ('VPM', 'vpm'),
    ('VRPM', 'vrpm'),
    ('References', 'refs'),
    # ('Code on GitHub', 'https://github.com/bgerazov/prosodeep'),
)
LINKS = (
('Code on GitHub', 'https://github.com/bgerazov/prosodeep'),
)
# MENUITEMS = (
#     ('Home', ''),
#     ('Project', 'project'),
#     ('PySFC', 'pysfc'),
#     ('WSFC', 'wsfc'),
#     ('VPM', 'vpm'),
#     ('VRPM', 'vrpm'),
#     ('References', 'refs'),
#     # ('Code on GitHub', 'https://github.com/bgerazov/prosodeep'),
# )

# THEME = 'simple'
# THEME = 'notmyidea'
# THEME = 'brutalist'
# THEME = 'gum'
# THEME = 'aboutwilson'
# THEME = 'pelican-blue'
# THEME = 'pelican-sober'
# THEME = 'resume'
# THEME = 'nice-blog'
# THEME = 'mg'
# THEME = 'tuxlite_tbs'
THEME = 'tuxlite_zf'
# THEME = 'twenty-html5up'
# THEME = '/home/vibe/pelican-themes/' + THEME
# Blogroll
# LINKS = (('Pelican', 'http://getpelican.com/'),
#          ('Python.org', 'http://python.org/'),
#          ('Jinja2', 'http://jinja.pocoo.org/'),
#          # ('You can modify those links in your config file', '#'),
#          )

# Social widget
# SOCIAL = (
#           # ('You can add links in your config file', '#'),
#           # ('Another social link', '#'),
#           )

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
