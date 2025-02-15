+++
title = 'Hacking F45 LionHeart: Using a Garmin HRM Instead'
date = 2025-02-15T12:55:16Z
tags = ['fitness']
+++

## Why I Joined F45

I recently joined F45 to lose some weight and force myself to do more weight training. I generally prefer running or cycling outside, so this seemed like a good way to mix things up.

F45 promotes its **LionHeart** heart rate monitor, which connects to a receiver in the studio and provides:

- Live heart rate data displayed on screens during the class
- Calories burned and LionHeart points (whatever those are)

Since I already use a Garmin Forerunner running watch, I wasn’t interested in paying extra for a proprietary device that offers minimal added benefit. However, I soon realised there was more to the system than I initially thought.

### The Strava Sync Let-down

F45 allows you to [link your account to Strava](https://support.strava.com/hc/en-us/articles/27745923005837-F45-and-Strava), automatically posting your workout along with the type of class it was. In theory, this should provide a more accurate calorie estimate because F45 knows what you did in class.

After linking my F45 account, I went to my first session expecting my heart rate data to sync to Strava. But when I checked the activity later - nothing. No HR data. 😢

### The Reddit Discovery: ANT+ Compatibility

A quick Google search revealed that LionHeart data is supposed to sync to Strava. Meaning, if I had a LionHeart monitor, my heart rate and calorie data would show up.

Digging deeper, I found a [Reddit post](https://www.reddit.com/r/f45/comments/1ceuypa/comment/l1l2zbu/) that revealed an interesting detail: **LionHeart just uses ANT+**. That means any ANT+ heart rate monitor - like my Garmin Dual HRM chest strap or my Garmin watch - should work. 🎉

## Setting Up My Garmin HRM With F45

I chose to use my Garmin HRM over my running watch as it is typically more accurate, especially when you’re moving your arms a lot like you do at F45.

F45 requires you to enter a **Device ID** in the app to link your heart rate monitor, which corresponds to the **ANT+ Sensor ID**. To find the Sensor ID of my Garmin HRM, I checked my Garmin Edge 530 which was already paired to it, which reported a value of `863436`. I entered this into the F45 app and eagerly awaited my data appearing on the class screens.

Unfortunately, it didn’t work as expected. When I went to my next class, a device showed up on the screen as `Guest#11486` but wasn’t linked to my account. 🤔

### The “Magic Number” Problem

Another [Reddit post](https://www.reddit.com/r/f45/comments/jw9vhl/comment/gd1mhu2/) mentioned a “magic number” that needed to be entered instead. This corresponds to the number after `Guest#`. But where did this number come from?

The issue stems from how [ANT+ assigns device addresses](https://forums.garmin.com/developer/connect-iq/f/discussion/317491/ant-device-numbers). Originally, ANT+ used **16-bit addresses**, but newer devices use **20-bit addresses**. Some systems (including F45) still expect a 16-bit number, meaning we need to convert our 20-bit Sensor ID to its 16-bit equivalent.

### Converting an ANT+ Sensor ID for F45

To calculate the correct Device ID for F45:

1. Convert the number from your Garmin device to **Hexadecimal**. **Example:** `863436` → `0xD2CCC`
2. Drop the first 4 bits (the first hexadecimal character). **Example:** `0xD2CCC` → `0x2CCC`
3. Convert back to **Decimal**.**Example:** `0x2CCC` → `11486`

Once I entered `11486` into the F45 app, my Garmin HRM was recognised when I to my next class, and my heart rate data was displayed correctly on the screens. 🎯

## Final Thoughts

It’s always satisfying to figure out _why_ something works rather than just following a vague fix. If you already own an ANT+ heart rate monitor, there’s no need to buy an F45 LionHeart — you can simply use this method to get it working.

Hopefully, this helps others looking to use their own devices with F45!
