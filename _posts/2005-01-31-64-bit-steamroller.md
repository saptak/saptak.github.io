---
author: Saptak Sen
date: '2005-01-31T12:34:00.001-07:00'
header_image_path: /assets/img/blog/headers/2005-01-31-64-bit-steamroller.jpg
image_credit: Photo by Unsplash
layout: post
modified_time: '2005-01-31T15:11:18.054-07:00'
tags:
- 64-bit
- windows
thumbnail_path: /assets/img/blog/thumbnails/2005-01-31-64-bit-steamroller.jpg
title: 64 bit Steamroller on our way
---

Over the weekend I installed and configured a new build [v. 1421] of the Windows XP x64 Edition on my Compaq Presario 3000 Laptop with a 64-bit processor (just in case you have not noticed it on the screenshot above).[  
![](http://photos1.blogger.com/img/98/1747/320/presario_r3000.jpg)](http://photos1.blogger.com/img/98/1747/640/presario_r3000.jpg)  
Compaq Presario R3000  

A question I get asked quite often:  
64-bit Keun?(Why 64-bit?)  

If you are running an enterprise information technology solution, one of the common strategies for you to adopt in order to scale-up (as opposed to scale-out) your solution would be to throw more memory at your box.  

With a 32-bit hardware you will hit the memory ceiling of 4GB = 2^32 bits (the maximum addressable memory space with piddly 32-bits), unless you unleash some tricks like AWE, PAE, etc to overcome the memory barrier. These tricks have their own overheads.  

I am sure you have heard of the saying "There's nothing called a free lunch".  

With 64-bit processors this ceiling just gets blown away, since now you have a maximum theoretical addressable memory space become 2^64 bits, which is 18446744073709551616 bits. This is about 18 billion GB, or 18 exabytes ([http://en.wikipedia.org/wiki/Exabyte](http://en.wikipedia.org/wiki/Exabyte)). Now hopefully you will not require more than 18EB of memory anytime soon. But, who can say ;-).  

The next question I get asked is, what am I doing with a 64-bit capable laptop. Am I planning to run all the enterprise solution of my company on my shining 64-bit laptop.  

No, my dear. The reason you and I (who are power users :-) offcourse) need a 64-bit laptop is because I may decide to take a shot at the GrammyAwards by creating some soul stirring music entirely on my laptop w/o visiting any recording studio. Check out the software: [http://www.cakewalk.com/x64/default.asp](http://www.cakewalk.com/x64/default.asp)  
Here are some example of music which has been entirely created on a computer. Check it out, I bet you will be pleasantly surprised at the quality of these music:  


  * [http://download.com/kishoresen](http://download.com/kishoresen)
  * [http://www.soundclick.com/bands/8/sensationmusic.htm](http://www.soundclick.com/bands/8/sensationmusic.htm)
  * [http://kishore.dmusic.com/](http://kishore.dmusic.com/)
  * [http://www.soundclick.com/sensation](http://www.soundclick.com/sensation)
  * [http://www.geocities.com/kisor_s/](http://www.geocities.com/kisor_s/)
Or, I might just decide to check out the latest and greatest first action shooter game which is so realistic and thereby so memory hungry that 2GB RAM is not adequate for the unreal experience. Check out how you can lay your hands on a free copy of 64-bit game "Far Cry" [http://www.amd.com/us-en/Processors/DevelopWithAMD/0,,30_2252_875_10543,00.html?redir=IEGFC01](http://www.amd.com/us-en/Processors/DevelopWithAMD/0,,30_2252_875_10543,00.html?redir=IEGFC01)  

Now for the real reason for my possessing this 64-bit hardware; I help Independent Software Vendors write efficient software on 64-bit Windows platform. So, this is my lab where I perform different experiments to check out my ideas before I present them to my audience.