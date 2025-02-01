import re
from pathlib import Path


# with open("docs/faq/index.md", 'r', encoding='utf-8') as working_file:
#     for line in working_file:
#         has_html = re.search(r".html", line)
#         if has_html:
#             has_http = re.search(r"https?:", line)
#             has_api = re.search(r"api/", line)
#             if not (has_http or has_api):
#                 processed_lines.append(re.sub(r".html", ".md", line))
#             else:
#                 processed_lines.append(line)
#         else:
#             processed_lines.append(line)


for file in Path('docs').rglob('*.md'):
    processed_lines = []

    with open(file, 'r', encoding='utf-8') as working_file:
        print("Processing {}".format(file))
        for line in working_file:
            has_html = re.search(r".html", line)
            if has_html:
                has_http =re.search(r"https?:", line)
                has_api = re.search(r"api/", line)
                if not (has_http or has_api):
                    processed_lines.append(re.sub(r".html", ".md", line))
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

    with open(file, 'w', encoding='utf-8') as writing_file:
        writing_file.writelines(processed_lines)




#


# recursively find all the md files

# find all the .html that don't start with http(s) or contain the text "api"
# replace html with md