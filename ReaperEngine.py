import json
from openai import OpenAI
from bs4 import BeautifulSoup
import re
import os
from datetime import datetime
import random

import concurrent.futures
import requests
import replicate

''' About the name...
I apologise for it sounding pretentious or whatever, but I dont care it sounds cool and cyberpunk-y(-ish)
and fits with the Dead Internet Theory theme of this little project
'''


class ReaperEngine:
    def __init__(self):
        # groq not local but so fast that the browsing feels quite normal
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1",
                             api_key="")
        self.model = "llama3-8b-8192"

        # ollama local and ok with gpu but nothing compared to groq
        # self.client = OpenAI(base_url="http://localhost:11434/v1/", api_key="Dead Internet")  # Ollama is pretty cool
        # self.model = "llama3"

        self.generate_images = False
        local_img_gen = True  # True for local image gen, False for replicate api

        # Not needed if local_img_gen = True
        os.environ['REPLICATE_API_TOKEN'] = ''

        self.internet_db = dict()  # TODO: Exporting this sounds like a good idea, losing all your pages when you kill the script kinda sucks ngl, also loading it is a thing too

        self.temperature = 0.5  # Crank up for goofier webpages (but probably less functional javascript, and it will probably error out a lot)
        self.max_tokens = 4096

        # 1. base prompt
        # 2. Image prompt thingy(you can make it generate links and alt for sdxl turbo)
        # 3. Output style CoT
        self.system_prompt = "You are an expert in creating realistic webpages. You do not create sample pages, instead you create webpages that are completely realistic and look as if they really existed on the web. During this process make sure the websites are quite long and include all needed details. You do not respond with anything but HTML, starting your messages with <!DOCTYPE html> and ending them with </html>. If a requested page is not a HTML document, for example a CSS or Javascript file, write that language instead of writing any HTML."

        self.pipe = None
        if self.generate_images:
            # Generate image prompt
            self.system_prompt += "If the requested page is instead an image file or other non-text resource, attempt to generate an appropriate resource. To make a image use the alt tag to ender a detailed description of the image for image generation. Try to include a few images on every page while making sure every image has a alt tag."
            if local_img_gen:
                import torch
                from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                base = "stabilityai/stable-diffusion-xl-base-1.0"
                repo = "ByteDance/SDXL-Lightning"
                ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

                # Load model.
                unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
                unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
                pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16,
                                                                 variant="fp16").to("cuda")

                # Ensure sampler uses "trailing" timesteps.
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config,
                                                                         timestep_spacing="trailing")
                self.generate_image = self.generate_image_local
            else:
                self.generate_image = self.generate_image_replicate
        else:
            # Dont generate image prompt
            self.system_prompt += " If the requested page is instead an image file or other non-text resource, attempt to generate an appropriate resource while avoiding using images. You use no images at all in your HTML, CSS or JS."

        # Output style prompt
        self.system_prompt += "\n\nThe response should have two parts to it **Planning Stage**, **HTML**. First planning indicated by **Planning Stage** in which you plan out any elements(try not to do anything to complex and stick with simple but modern design. Try to avoid gradients as they often have problems) that are needed as well as any text. This should be quite detailed to create a modern realistic webpage with all required things. Make sure you plan enough to fill the entire website adequately so dont cut it short. This means you leave no placeholders and fully plan out all text. You dont only plan, but also write all text that should be present later. The text should have at least 500 words that will be on the webpage depending on the type. You do not create any code during this stage. The next stage is writing the html. To do this simpy start with <!DOCTYPE html> and generate the needed code based of the previous plan following all instructions. You have to generate all parts, none are optional **Planning Stage**, **HTML**. Stick to a dark mode color Theme and include lots of links throughout the page not just at the end but for all important things."

        self.search_system_prompt = "Create at least 10 different results for the query from the user. Your answer should have this format. 1. [TITLE:INSERT_TITLE_HERE] of course replace INSERT_TITLE_HERE with the actual title of the website. It should be relatively short but informative just like google results. 2. [DESCRIPTION:INSERT_DESCRIPTION_HERE] it should in a few words convey the most important info on the webpage. 3. [LINK:INSERT_LINK_HERE] It should have all needed info about the page it links to so dont shorten it at all. Here is a example of how a part of your response could look like. [TITLE:The Tree Encyclopedia: A Comprehensive Guide] \n [DESCRIPTION:Explore the world of trees, from their evolution to their uses, and get information on over 1,000 species.]\n [LINK:https://www.treecyclopedia.org/guide/history+evolution+uses+species] then just start with the next result. Repeat this for all requested results. Of course dont include any placeholders and instead replace them with proper information. Make sure you stick to the format at all times without exceptions."

    def generate_image_local(self, prompt, num_outputs=1):
        self.pipe(prompt, negative_prompt="worst quality, low quality", num_inference_steps=4,
                  guidance_scale=0).images[0].save("output.png")
        # TODO: Implement local image generation logic here (maybe both sdxl turbo and lightning)
        return ["Local image generation not implemented"]

    def generate_image_replicate(self, prompt, num_outputs=1):
        # Call replicate api to generate image
        output = replicate.run(
            "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
            input={
                "width": 1024,
                "height": 1024,
                "prompt": prompt,
                "scheduler": "K_EULER",
                "num_outputs": num_outputs,
                "guidance_scale": 0,
                "negative_prompt": "worst quality, low quality",
                "num_inference_steps": 4
            }
        )

        # Download images from replicate
        images = []
        for url in output:
            # print(f"Downloading image from {url}")  # Debug output
            response = requests.get(url)
            if response.status_code == 200:
                images.append(response.content)
                # print(f"Downloaded {len(response.content)} bytes")  # Debug output
            else:
                raise Exception(f"Failed to download image from {url} with status code {response.status_code}")

        return images

    def _format_page(self, dirty_html):
        # Teensy function to sanitize links on the page, so they link to the root of the server
        # Also to get rid of any http(s), this will help make the link database more consistent
        soup = BeautifulSoup(dirty_html, "html.parser")
        for a in soup.find_all("a"):
            if "mailto:" in a["href"]:
                continue
            a["href"] = a["href"].replace("http://", "")
            a["href"] = a["href"].replace("https://", "")
            if not a["href"].startswith("http://127.0.0.1:5000/"):
                a["href"] = "http://127.0.0.1:5000/" + a["href"]

        # Create a new 'a' tag for the Home button
        home_button = soup.new_tag("a", href="/")
        home_button.string = "Home"

        # Insert the Home button at the start of the body element
        body = soup.body
        if body:
            body.insert(0, home_button)

        html = str(soup)
        return html

    def insert_images(self, html, output_dir="static/images"):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all instances of <img...> in the HTML
        image_tags = re.findall(r'<img.*?>', html)

        # Array to store prompts, original HTML locations, and final file names
        image_data = []

        for tag in image_tags:
            # Extract the alt text from the tag
            alt_match = re.search(r'alt="([^"]*)"', tag)
            if alt_match:
                prompt = alt_match.group(1).strip()
                file_name = f"{output_dir}/image_{random.randint(1, 1000000000)}.png"
                image_data.append((prompt, tag, file_name))
            else:
                # Remove image from HTML if no alt text
                html = html.replace(tag, '')

        def generate_and_save_image(data):
            prompt, tag, file_name = data
            print(data)
            images = self.generate_image(prompt)
            image = images[0]  # Generate the image
            with open(file_name, "wb") as img_file:
                img_file.write(image)
            return tag, file_name, prompt

        # Use ThreadPoolExecutor to generate images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(generate_and_save_image, image_data))

        # Update the HTML with new image paths
        for tag, file_name, prompt in results:
            # Make the file_name relative to the static folder
            relative_file_name = file_name.replace(output_dir, '').lstrip('/')
            new_tag = f'<img src="{{{{ url_for(\'static\', filename=\'images/{relative_file_name}\') }}}}" alt="{prompt}">'
            html = html.replace(tag, new_tag)
        print(html)
        return html

    def separate_html_parts(self, html):
        doctype_index = html.lower().find('<!doctype html>')
        if doctype_index == -1:
            raise ValueError("HTML does not start with <!DOCTYPE html>")
        return html[doctype_index:]

    def remove_images(self, html):
        # Remove img tags
        html = re.sub(r'<img.*?>', '', html)

        # Remove image links (e.g. <a href="image.jpg">)
        html = re.sub(r'<a.*?href="[^"]+\.(jpg|jpeg|png|gif|bmp)".*?</a>', '', html)

        # Remove background-image CSS properties
        html = re.sub(r'background-image:\s*url\([^)]+\);', '', html)

        # Remove image references in CSS styles (e.g. style="background-image: url('image.jpg')")
        html = re.sub(r'style="[^"]*background-image:\s*url\([^"]+\)[^"]*"', '', html)

        # Remove image references in HTML attributes (e.g. <div data-src="image.jpg">)
        html = re.sub(r'\b(data-src|src|lowsrc)="[^"]+\.(jpg|jpeg|png|gif|bmp)"', '', html)

        return html

    def get_index(self):  # html for start search page
        return """
        <!DOCTYPE html>
<html>
<head>
    <title>Dead Internet</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #222;
            color: #fff;
        }
        .container {
            max-width: 600px;
            margin: 100px auto;
            text-align: center;
        }
        .logo {
            font-size: 36px;
            font-weight: bold;
            color: #fff;
        }
        .search-form {
            margin-top: 40px;
        }
        .search-input {
            width: 100%;
            height: 50px;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            background-color: #333;
            color: #fff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .search-button {
            height: 50px;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            background-color: #4285f4;
            color: #fff;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .search-button:hover {
            background-color: #357ae8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">Dead Internet</div>
        <form class="search-form" action="/">
            <input class="search-input" type="text" name="query" placeholder="Search the Dead Internet">
            <input class="search-button" type="submit" value="Search">
        </form>
    </div>
</body>
</html>
        """

    def get_page(self, url, path):
        # Return already generated page if already generated page
        generated_page = self.internet_db.get(url, {}).get(path)
        if generated_page:
            return generated_page

        current_time = datetime.now()
        # Format date and time
        formatted_date = current_time.strftime("%A, %d %B %Y %H:%M:%S")

        # Append to system prompt
        system_prompt = self.system_prompt + f"Today is the {formatted_date}"

        # Construct the basic prompt
        prompt = f"Give me a classic geocities-style webpage from the fictional site of '{url}' at the resource path of '{path}'. Make sure all links generated either link to an external website, or if they link to another resource on the current website have the current url prepended ({url}) to them. For example if a link on the page has the href of 'help' or '/help', it should be replaced with '{url}/path'. All your links must use absolute paths, do not shorten anything. Make the page look nice and unique using internal CSS stylesheets, don't make the pages look boring or generic."
        # TODO: I wanna add all other pages to the prompt so the next pages generated resemble them, but since Llama 3 is only 8k context I hesitate to do so --> summary of previous page

        # Add other pages to the prompt if they exist
        if url in self.internet_db and len(self.internet_db[url]) > 1:
            pass

        # Generate the page
        generated_page_completion = self.client.chat.completions.create(messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Get and format the page
        generated_page = generated_page_completion.choices[0].message.content
        with open("templates/curpage.html", "w+") as f:
            f.write(generated_page)

        # separate the different parts of the answer before further processing
        generated_page = self.separate_html_parts(generated_page)

        if self.generate_images:
            # find alt tags and then batch request images
            generated_page = self.insert_images(generated_page)
        else:
            # remove images as the browser will otherwise try to open pages that will never get seen --> useless traffic
            generated_page = self.remove_images(generated_page)

        # clean up links and add home button
        generated_page = self._format_page(generated_page)

        # Add the page to the database
        if url not in self.internet_db:
            self.internet_db[url] = {}

        self.internet_db[url][path] = generated_page

        return generated_page

    def get_search(self, query):  # Generate search results
        current_time = datetime.now()
        # Format date and time
        formatted_date = current_time.strftime("%A, %d %B %Y %H:%M:%S")

        # Append to system prompt
        search_system_prompt = self.search_system_prompt + f"Today is the {formatted_date}"

        # Generate possible search results
        search_page_sites = self.client.chat.completions.create(messages=[
            {
                "role": "system",
                "content": search_system_prompt
            },
            {
                "role": "user",
                "content": f"Generate the search results for a fictitious search engine where the search query is '{query}'. Please include at least 10 results to different fictitious websites that relate to the query. Each search result will have a link to the site it referring to. Make sure each fictitious website has a unique and somewhat creative URL with lots of info. Don't mention that the results are fictitious."
            }],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Separate the possible search results for use by create search site
        site_array = self.separate_sites(search_page_sites.choices[0].message.content)

        # Create the html for the search site
        html = self.create_search_site(site_array, query)
        html = self._format_page(html)
        return html

    def create_search_site(self, site_array, query):
        """
        :param site_array: Array of sites where each site is a list [title, description, link]
        :param query: Search query string
        :return: HTML string representing the search results page

        Create a modern search results page with a dark mode style and rounded edges.
        """

        # Start the HTML output with a modern dark mode style
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>Search Results</title>
        <style>
        body {{
            background-color: #1c1c1c;
            color: #e6e6e6;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}

        .search-header {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .search-results {{
            list-style-type: none;
            padding: 0;
        }}

        .search-result {{
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .search-result a {{
            color: #e6e6e6;
            text-decoration: none;
        }}

        .search-result a:hover {{
            text-decoration: underline;
        }}

        .search-result .description {{
            font-size: 14px;
            color: #b3b3b3;
            margin-top: 5px;
        }}
        </style>
        </head>
        <body>
        <div class="container">
        <div class="search-header">
            <h1>Searched for: {query}</h1>
        </div>
        <ul class="search-results">
        """

        # Loop through the site array and create list items for each site
        for site in site_array:
            title, description, link = site
            html += f"""
            <li class="search-result">
                <a href="{link}"><h2>{title}</h2></a>
                <p class="description">{description}</p>
            </li>
            """

        # End the HTML output
        html += """
        </ul>
        </div>
        </body>
        </html>
        """

        return html

    def separate_sites(self, raw_output):
        """
        :param raw_output: A string containing raw site data
        :return: An array that separates this for each site

        The raw data has this structure:
        [TITEL or TITLE:INSERT_TITLE_HERE]
        [DESCRIPTION:INSERT_DESCRIPTION_HERE]
        [LINK:INSERT_LINK_HERE]

        It repeats this structure.

        The output is an array of arrays, each containing the title, description, and link.
        """

        # Define the regex pattern allowing both "TITEL" and "TITLE"
        pattern = re.compile(r'\[(?:TITEL|TITLE):(.*?)]\s*\[DESCRIPTION:(.*?)]\s*\[LINK:(.*?)]', re.DOTALL)

        # Find all matches in the raw_output
        matches = pattern.findall(raw_output)

        # Create an array of sites
        site_array = [list(match) for match in matches]

        # if site_array empty throw error
        if not site_array:
            raise ValueError("Site array empty: No matches found in the raw output.")

        return site_array

    def export_internet(self, filename="internet.json"):
        json.dump(self.internet_db, open(filename, "w+"))
        russells = "Russell: I'm reading it here on my computer. I downloaded the internet before the war.\n"
        russells += "Alyx: You downloaded the entire internet.\n"
        russells += "Russell: Ehh, most of it.\n"
        russells += "Alyx: Nice.\n"
        russells += "Russell: Yeah, yeah it is."
        return russells
