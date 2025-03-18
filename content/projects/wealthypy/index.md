+++
title = 'Wealthypy'
date = 2025-03-18T13:10:32Z
tags = ['investing']
+++

For a long time, I put off sorting out my pension because the whole concept felt overwhelming. It sounded quite complicated, filled with technical jargon, and I wasn't sure where to start. But as I dug into it, I discovered that a pension is really just a **tax-efficient investment wrapper**, and not something fundamentally different from investing in general. Once I understood that, it became much less intimidating. Once I chose a suitable (and cheap) SIPP platform [^1], "all" I need to do was decide what to invest in.

## Discovering rational investing

Initially, I assumed that successful investing meant picking individual stocks, keeping up with market trends, and making the right bets. I didn't feel like I had any expertise in the world of finance, and it sounded like a full-time job.

Then I read [_Investing Demystified_](https://www.amazon.co.uk/Investing-Demystified-Speculation-Sleepless-Financial/dp/0273781340) by Lars Kroijer, which introduced me to a much simpler approach to investing requiring little effort. Instead of trying to beat the market, a rational investor takes a more passive approach by investing in a cheap global equity index fund to generate returns. Essentially you allow the market to choose the optimal portfolio of shares for you.

You can then balance this out with some government bonds as a minimal risk asset, so you can tune your risk level. This passive approach takes the complexity out of investing and gives you a solid foundation for long-term growth.

## Building a portfolio simulation tool

Once I decided to go with passive investing, the next challenge was tuning the portfolio to match my risk level. The right balance between equities and bonds depends on your individual needs; eg time horizon, risk tolerance, and financial goals.

I struggled to find any freely available tools to help with this, so decided to build my own. Using the model presented in _Investing Demystified_, I created a simple [Streamlit dashboard](https://wealthypy.streamlit.app/) to simulate portfolio growth over time.

{{< github repo="alxhslm/wealthypy" >}}

### Features of the tool

- Set a **starting amount** and **monthly contributions**, which can vary over time.

- Adjust the **equity and bond returns**, as well as the **equity-bond split**, which can also vary over time.

- Simulate how your portfolio might grow while accounting for **equity volatility** using a Monte Carlo-like approach.

This has helped me choose how much to contribute to my pension, fine-tune the portfolio to my needs, as well as gain confidence in my overall financial plan. I hope it can help others do the same.

At some point I hope to further improve the tool to incorporate additional assets like corporate bonds (also recommended in the book) and evaluate your investment strategy on historical data as well.

[^1]: I would recommend looking at [MSE](https://www.moneysavingexpert.com/savings/cheap-sipps/) as it can depends a lot on how much money you have
