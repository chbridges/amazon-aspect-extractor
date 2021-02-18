import os


from torch.cuda import is_available
from utils.application import run_app
from pipeline import Pipeline

if __name__ == "__main__":
    # input_url = ['https://www.amazon.com/-/de/dp/B07RF1XD36/ref=lp_16225009011_1_6',
    #             'https://www.amazon.com/dp/B08JQKMFFB/ref=sspa_dk_detail_2?psc=1&pd_rd_i=B08JQKMFFB&pd_rd_w=5AdCg' +
    #             '&pf_rd_p=45e679f6-d55f-4626-99ea-f1ec7720af94&pd_rd_wg=bWbE5&pf_rd_r=HJV72D1QHGE2XJ8QJBV0&pd_rd_r' +
    #             '=b3a4c265-2d13-454f-a385-3ad0a71737eb&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzN1NSWjVRTFFINUFNJmVuY3J5cHR' +
    #             'lZElkPUEwMjY3OTk2MUQ5ODYwVU4zNlhBVCZlbmNyeXB0ZWRBZElkPUEwMzMwMjc2M1VQMVJXVllMVVpGJndpZGdldE5hbWU9c' +
    #             '3BfZGV0YWlsJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==',
    #             'https://www.amazon.com/product-reviews/B08KH53NKR/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar' +
    #             '=all_stars&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar'
    #             ]

    device = "cuda" if is_available() else "cpu"

    P = Pipeline(output_size=3,
            model_name="laptops_best",
            device=device,
            normalize_output=False,
            n_layers=1,
            embedding_dim=300,
            hidden_dim=128,
            dropout=0.35,
            bidirectional=False,
    )
    run_app(P)
