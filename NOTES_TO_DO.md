## Notes:
* Most people are going to want to start with `mkdocs server --dirty` so they don't rebuild the whole site every time

* Do the search and replace for 404s separately between the jupyter notebooks and the markdown pages. THe python notebooks are using https: to link to internal pages since we want them to still work when downloaded

## TODO

1. If a md file.md is given a link in the nav, rather than generaing the file.html  - it generates a file/index.html. Is there some way to turn off this behavior. This behavior will break even more links than necessary

2. Why does the docs home page link images from the main corporate site?

3. Decide if getting starteds are actually notebooks