import src.utils.reviewextractor as rw
import json

if __name__ == '__main__':
    urls = ['https://www.amazon.com/-/de/dp/B081V6W99V/ref=sr_1_3?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&dchild=1&keywords=laptop&qid=1612430596&sr=8-3',
            'https://www.amazon.com/-/de/dp/B078H42W49/ref=sr_1_17?__mk_de_DE=ÅMÅŽÕÑ&dchild=1&keywords=MacBook&qid=1612430532&sr=8-17',
            'https://www.amazon.com/-/de/dp/B084SMMH96/ref=sr_1_4?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&dchild=1&keywords=laptop&qid=1612430569&sr=8-4',
            'https://www.amazon.com/-/de/dp/B00VQR7MVQ/ref=sr_1_6?__mk_de_DE=ÅMÅŽÕÑ&dchild=1&keywords=MacBook&qid=1612430532&sr=8-6',
            'https://www.amazon.com/dp/B0821ZJBF8/ref=sr_1_1_sspa?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&dchild=1&keywords=laptop&qid=1612430536&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUFXTk9TTFJUQzFLTkUmZW5jcnlwdGVkSWQ9QTA1MDYyNzFEVU9DOEY4SkFZRTkmZW5jcnlwdGVkQWRJZD1BMDYwMzE0NjNTV0hQM1kxWThNNFgmd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl',
            'https://www.amazon.com/-/de/dp/B081FZV45H/ref=sr_1_5?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=29Z5XH719H3E4&dchild=1&keywords=macbook+pro&qid=1612430431&sprefix=Mac%2Caps%2C245&sr=8-5'
            ]
    data = rw.extract_reviews_for_products(urls, 1, 6)
    data = [item for sublist in data for item in sublist]
    json_string = json.dumps(data)
    f = open("review_export.json", "w")
    f.write(json_string)
    f.close()

