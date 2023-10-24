import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

# scrapy runspider langchain_spider.py -o items.csv -t csv
class PythonLangchainSpider(CrawlSpider):
    name = 'python_langchain'
    allowed_domains = ['python.langchain.com']
    start_urls = ['https://python.langchain.com/docs/get_started']

    # Rules for link extraction
    rules = (
        # Rule to match links embedded within /docs and follow them.
        Rule(LinkExtractor(allow=('/docs',)), follow=True, callback='parse_item'),
    )

    def parse_item(self, response):

        url = response.url
        title = response.css('h1::text').get()
        content = ""


        for span in response.css('span'):
            span_text = "\n".join(span.css('::text').getall())
            if span_text: 
                content += "[CODE] " + span_text + "\n"

        content += "\n"

        # Add main content sections      
        for paragraph in response.css('p'):
            paragraph_text = "\n\n".join(paragraph.css('::text').getall())  
            content += paragraph_text + "\n\n"
        

        yield {
        'source': url,
        'title': title,
        'content': content
        }
