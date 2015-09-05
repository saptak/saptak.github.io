---
layout: post
title: Data dependency concerns in parallel computing
date: '2007-08-29T12:34:00.001-07:00'
author: Saptak Sen
tags:
- HPC
- Parallel Computing
modified_time: '2007-08-29T15:11:18.054-07:00'
---

The most severe bottlenecks in high performance systems in majority cases results from I/O operations. To buffer I/O or other slow accesses, engineers devised cache.   

So, what is a cache, how does it work, and what should we know to intelligently program? According to an English dictionary, it is a safe place to hide things. In the context of computer memory, it can be described as a place of storage that is close by.   

Since bulk storage for data is usually relatively far from CPU, the principle of data locality encourages having a fast data access for data being used, hence likely to be used next, that is, close by and quickly accessible.   

Then writing data from memory cache is again a problem, even though it occurs only roughly one-fourth as often as reading data. In writing modified data back into memory, these data cannot be overwritten onto old data which should be subsequently used for processes issued earlier.   

Conversely, if the programming language ordering rules dictate that an updated variable is to be e not used for the next step, it is clear this variable must be safely stored before it is used. Since bulk memory is usually far away from the CPU, why write the data all the way back to their rightful memory locations if we want them for a subsequent step to be computed very soon? Two strategies are in use.  

  1. A ‘write through' strategy automatically writes back to memory any modified variables in cache. A copy of data is kept in cache for subsequent use. This copy might be written over by other data mapped to the same location in cache without worry. A subsequent cache miss on the written through data will be assured to fetch valid data from memory because the data are freshly updated on each write. 
  2. A ‘write back' strategy skips the writing to memory until: 
    1. a subsequent read tries to replace a cache block which has been modified, or 
    2. these cache resident data are modified by the CPU. 

These two situations are more or less the same: cache resident data are not written back to memory until some process tries to modify them. Otherwise, the modification would write over computed information before it is saved.   

It is well known that certain processes, I/O and multi-threading, for example, want it both ways. In consequence, modern cache designs often permit both write-through and write-back modes. Which mode is used may be controlled by the program.
