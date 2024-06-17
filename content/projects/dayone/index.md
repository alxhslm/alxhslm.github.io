+++
title = 'Day One Importer'
date = 2021-01-03T15:36:00Z
tags = ['journal']
+++

I started journalling a few years ago, and I prefer to do this digitally because I appreciate the improved discoverability when compared to a paper notebook. Initially I used the [Journey](https://journey.cloud/) app because:

- It was the only cross-platform option at the time (via the web app)
- It was a lot cheaper than the alternatives (since there was a one-off payment option)

Once I switched to using primarily Apple devices, I decided to take a look again at the different options. Although I liked Journey, it did not have the same polish as [Day One](https://dayoneapp.com/), so I made the switch. I had around 2 years of journal entries to migrate, but unfortunately Day One only offer [Journey import on Android](https://dayoneapp.com/guides/settings/importing-data-to-day-one/#Android). I really dislike vendor lock-in like this, and it was particularly frustrating when the developer has clearly already done most of the work.

Therefore I decided to create my own tool to perform the conversion automatically. Fortunately Day One and Journey both store their entries in JSON format under the hood. Once I reverse engineered the schema with some trial and error, the conversion logic was relatively simple. The code is open source and available at the repo below.

{{< github repo="alxhslm/journey2dayone" >}}
