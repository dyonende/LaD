from bs4 import BeautifulSoup
from newspaper import Article
from tldextract import extract
import pandas as pd
import requests
import json
import os
import re
import numpy as np
import sys

# Global JSON object with settings on how to select specific elements for each news agency website
mt_settings={
    'tvm': {
        'query': 'https://www.tvm.com.mt/mt/news/page/$page$/?phrase=$keywords$', 
        'select': {
            'html': '.archive-news-box',
            'title': '.archive-news-box div.title', 
            'url': '.archive-news-box > a', 
            'category': '.archive-news-box .category-date .category', 
            'tags': '.archive-news-box .categories-tags .category', 
            'date': '.article-meta .entry-date.published', 
            'authors': '.main-news-block .article-meta > div', 
            'content': '.main-news-block .content .text p'
            }
        },
    'net': {
        'query': 'https://netnews.com.mt/page/$page$/?s=$keywords$', 
        'select': {
            'html': '.herald-site-content .herald-posts .post.type-post',
            'title': '.herald-main-content .entry-title a', 
            'url': '.herald-main-content .entry-title a', 
            'category': '.herald-main-content .meta-category a', 
            'tags': '', 
            'date': '.herald-single .herald-date span', 
            'authors': '.herald-single .herald-author-name', 
            'content': '.herald-single .entry-content'
            }
        },
    'one': {
        'query': 'https://www.one.com.mt/news/page/$page$/?s=$keywords$', 
        'select': {
            'html': '#main .results-container article',
            'title': '.entry-title a', 
            'url': '.entry-title a', 
            'category': '.entry-category a', 
            'tags': '', 
            'date': '.meta-date .published', 
            'authors': '.meta-author > span > span', 
            'content': 'article.post .entry-content p'
            }
        },
}

def check_path(language):
    '''
    create language folder if it does not exist yet
    '''
    if os.path.exists(path_to_file) == False:
        os.mkdir(path_to_file)
        print("created folder", path_to_file)
    if os.path.exists(path_to_file+language+"/") == False:
        os.mkdir(path_to_file+language)
        print("created folder", path_to_file+language)

def get_links(keyword, language):
    '''
    get all links to news articles listed on Google News
    for the given keyword
    '''
    
    if len(keyword.split())>1:
        keyword = "+".join(keyword.split())
        print("keyword was changed to", keyword, "to remove whitespace")
        
    domain = "https://news.google.com"
    url = domain + "/search?q=" + keyword + "&hl=" + language
    
    #use a set to prevent double links
    links = set()

    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    articles = soup.find_all('article')
    for article in articles:
        for link in article.find_all('a'):
            try:
                article_link = (domain+link.get('href')[1:]) #remove . from relative link and add to domain to create absolute link
                r = requests.get(article_link) 
                links.add(r.url)
            except KeyboardInterrupt:
                sys.exit()
            except:
                pass
            
    print(len(links), "links extracted from", domain, "on topic", keyword)
    return links

def extract_content(content):
    '''
    extracts the content from the website page, while removing any new lines and tabs.
    '''
    result = ""
    if content != []:
        result = ", ".join([x.text for x in content]).replace("\n", "").replace("\t", "")
    return result

def url_to_html(url):
    '''
    Scrapes the html content from a web page.
    Takes a URL string as input and returns an html object.
    (code from Lisa Beinborn)
    '''
    res = requests.get(url, headers={"User-Agent": "XY"})
    html = res.text
    parser_content = BeautifulSoup(html, 'html5lib')
    return parser_content


def scrape_articles_mt(keywords, language, max_articles):
    check_path(language)
    write_dir = path_to_file + language + "/"
    outfile = "articles_"+language+".tsv"

    with open(write_dir+outfile, "w") as f:
        all_article_count = 0
        # write header
        f.write("website\tdate\ttitle\tlink\ttext\n")
        # for all news agencies
        for news_agency in mt_settings:
            article_count = 0 #article count per news agency
            print('::: ',news_agency, ' :::')
            # search all pages related to the keyword
            page = 0
            # loop until max_articles articles have been scraped (overhead of articles, in case some articles are corrupted or unusable)
            # (loop will keep going until all articles on the current search page has been scraped)
            while article_count < max_articles:
                try:
                    page += 1
                    # replace the keyholders $page$ and $keywords$ with the respective variables
                    query = mt_settings[news_agency]['query'].replace('$page$', str(page)).replace('$keywords$', keywords)
                    # request HTML content and select the search result sections
                    html = url_to_html(query)
                    posts = html.select(mt_settings[news_agency]['select']['html'])

                    # check if next page exists and scrape
                    if requests.get(query).status_code != 404:
                        print("page: ", page)
                        # for each post, collect title, url, category, tags, date, author(s) and content
                        for i, post in enumerate(posts):
                            title = extract_content(post.select(mt_settings[news_agency]['select']['title']))            # title
                            url = post.select(mt_settings[news_agency]['select']['url'])[0]["href"]                      # url
                            category = extract_content(post.select(mt_settings[news_agency]['select']['category']))      # category
                            # categories only exist for "tvm", in this case
                            tags = ""
                            if news_agency == 'tvm':
                                tags = extract_content(post.select(mt_settings[news_agency]['select']['tags']))          # tags

                            # get date, authors and content
                            article_html = url_to_html(url)
                            date = extract_content(article_html.select(mt_settings[news_agency]['select']['date']))          # date
                            authors = extract_content(article_html.select(mt_settings[news_agency]['select']['authors']))    # author
                            content = article_html.select(mt_settings[news_agency]['select']['content'])
                            content = " ".join([x.text for x in content])                                                 # content
                            # if content retreived is smaller than 10 characters (could be white space),
                            # ignore articles and go to the next one
                            if not content or len(content) < 10:
                                break

                            article_count += 1
                            all_article_count += 1
                            print(news_agency,"_article: ", article_count, " --- total articles: ", all_article_count)
                            print("\n".join([news_agency, date, title, url]))
                            print('---')

                            # write row to file
                            output = "\t".join([news_agency, date, title, url, content])
                            f.write(output +"\n")
                    else:
                        break
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    pass

            print('===')


    # Open output file and filter and clean data

    # read in abortion worksheet
    df = pd.read_csv(write_dir+outfile, sep='\t')
    # replace white space with NaNs, and then fill any NaNs with an empty string
    df['text'].replace(r'/\s/', np.nan, inplace=True, regex=True)
    df = df.fillna("")

    # filter out news articles unrelated to "abortion"
    df_abortion = df['title'].str.contains('abort')
    df = df[df_abortion]
    # select only a subset of a 100 articles
    #random sample to make sure not only 1 news agency is represented
    df_subset = df.sample(max_articles, random_state=1)

    # rewrite subset
    df_subset.to_csv(write_dir+outfile, sep='\t', index=False)


def scrape_articles_g_news(keyword, language, max_articles):
    '''
    search, scrape and save articles about the keyword
    '''
    
    print(f"getting links from Google News for topic {keyword} in language {language}")
    links = get_links(keyword, language)
    check_path(language)
    counter=1
    print("scraping news articles")
    
    for link in links:
        url = link
        result = dict()
        try:
            article = Article(url, language=language)
            article.download()
            article.parse()
            result["link"] = url
            result["date"] = str(article.publish_date)
            #https://stackoverflow.com/questions/44113335/extract-domain-from-url-in-python
            tsd, td, tsu = extract(url) #get url domain
            result["website"] = td
            result["text"] = article.text

            # write as .json file batch
            filename = language + "_" + keyword + '_' + str(counter) + ".json"
            with open(path_to_file + language + "/" +filename, 'w', encoding="utf-8") as outfile:
                json.dump(result, outfile, ensure_ascii=False)
            counter+=1

        except KeyboardInterrupt:
            return
        except:
            pass
        
        if counter > max_articles: #sufficient articles downloaded
            print("\ndone")
            break

# path to where data will be stored
path_to_file = input("provide path to store data (use './'  for current directory): ")
if path_to_file == "":
    path_to_file = "./"
if path_to_file[-1] != "/":
    path_to_file = path_to_file + "/"
    
scrape_articles_mt("abort", "mt", 100)
scrape_articles_g_news("abtreibung", "de", 100)
scrape_articles_g_news("abortus", "nl", 100)
