#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
from pathlib import Path

AUTHOR = 'gerazov'
SITENAME = 'ProsoDeep'
SITEURL = ''
HOME = str(Path.home())

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# DISPLAY_PAGES_ON_MENU = False
DISPLAY_CATEGORIES_ON_MENU = False
STATIC_PATHS = ['images']
TYPOGRIFY = True

# for a no blog website
TAGS_SAVE_AS = ''
TAG_SAVE_AS = ''
AUTHOR_SAVE_AS = ''
AUTHORS_SAVE_AS = ''
CATEGORY_SAVE_AS = ''
CATEGORIES_SAVE_AS = ''
ARTICLE_SAVE_AS = ''
ARTICLES_SAVE_AS = ''
YEAR_ARCHIVE_SAVE_AS = ''
MONTH_ARCHIVE_SAVE_AS = ''
DAY_ARCHIVE_SAVE_AS = ''

INCLUDE_TITLE = False
SITESUBTITLE = False
# GitHub  - fix this?
GITHUB_URL = 'https://github.com/bgerazov/prosodeep'
GITHUB_POSITION = 'right'

PAGES = (
    ('Home', '', 'index'),
    ('Project', 'project', 'project'),
    ('PySFC', 'pysfc', 'pysfc'),
    ('WSFC', 'wsfc', 'wsfc'),
    ('VPM', 'vpm', 'vpm'),
    ('VRPM', 'vrpm', 'vrpm'),
    ('Code', 'code', 'code'),
    ('Publications', 'pubs', 'pubs'),
)
# LINKS = (
# ('Code', 'https://github.com/bgerazov/prosodeep'),
# )

# global metadata to all the contents
DEFAULT_METADATA = {'Acknowledgement': 'Horizon 2020 Marie Skłodowska-Curie grant agreement No 745802'}
# code blocks with line numbers
PYGMENTS_RST_OPTIONS = {'linenos': 'table'}

# pelican folder
PELICAN_FOLDER = HOME + '/_pelican/'
THEME_PATHS = 'pelican_themes/'
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
# THEME = 'twenty-html5up'
# THEME = '~/pelican/pelican-themes/' + THEME
THEME = 'my_tuxlite_zf'

# plugins
PLUGIN_PATHS = [PELICAN_FOLDER + 'pelican-plugins']
PLUGINS = ['pelican-toc', 'render_math']
TOC = {
    'TOC_HEADERS'       : 'h[1-6]', # What headers should be included in
                                     # the generated toc
                                     # Expected format is a regular expression

    'TOC_RUN'           : 'true',    # Default value for toc generation,
                                     # if it does not evaluate
                                     # to 'true' no toc will be generated

    'TOC_INCLUDE_TITLE': 'false',     # If 'true' include title in toc
}
# MARKDOWN = {
#   'extension_configs': {
#     'markdown.extensions.toc': {
#       'title': 'Table of contents:'
#     },
#     'markdown.extensions.codehilite': {'css_class': 'highlight'},
#     'markdown.extensions.extra': {},
#     'markdown.extensions.meta': {},
#   },
#   'output_format': 'html5',
# }
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
