---
author: Saptak Sen
date: '2016-03-11T12:34:00.001-07:00'
header_image_path: /assets/img/blog/headers/2016-03-11-data-structures-in-python.jpg
layout: post
modified_time: '2016-03-11T12:34:00.001-07:00'
tags:
- Data Structure
- Python
thumbnail_path: /assets/img/blog/thumbnails/2016-03-11-data-structures-in-python.jpg
title: Data Structures in Python
---

In this post we will explore Data types in Python. The important thing about data structures in that you organize data to make certain things efficient.

Most Python programmers only use lists. A list allows you to store any number of information that has certain a sequence from the start of the list to the end of the list. You can search through it, you can insert, you can remove and you can iterate over all the elements.

A list is very generic structure with homogenous elements because the designers of the programming language does not know how we are going to use it. For example, lists of strings, lists of integers etc.

But, sometimes you need a specific collection with a specific behavior that you want. One such collection is a stack.

This collection is called a stack because when we consider a stack of paper sheets, the last sheet that is put on the stack is the one which is picked up first or in other words we process the collection of values in Last-In First-Out. So the operations are:

- Push value onto stack
- Pop and retrieve most recently added values
- Check if empty

We can emulate the operations on a list, but a stack is very different than a list, for example you never want to sort a stack as the items need to keep the order in which it has been pushed.

Lists are great for small amount of data, but as we scale to 10s of thousand or 100s of thousand elements we need a data structure that scales better.

A tuple looks exactly like a list except the elements are heterogenous and