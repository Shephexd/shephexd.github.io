---
layout: post
published: True
title: Data communications
excerpt: Introduction Data communications.
categories: theory
tags: 
- definition
- network
---

#Computer Network
##Basic definition
Routing
: When use network, It is route setting to send specific packet to other places. 


##Network model
###seven layers of the OSI model
1. Physical Layer
	- consist of hardware communication.
	- It like 0101110101010001, just binary data.
2. Data Link Layer
	- Flow control
3. Network Layer
	- Routing function to send packet via nodes.
	- IP protocol
4. Transport Layer
	- ,
	- TCP Protocol
5. Session Layer
	- ,
6. Presentation Layer
	- ,
7. Application Layer
	- data made by user to send other devices using network.

###TCP/IP Model
1 Physical layer (OSI 1,2 layers) 
: protocol defined by the underlying networks(host to network)

2 Network layer (OSI 3 layer)
: Use logical address
: IP, ICMP

3 Transport layer(OSI 4 layer) 
: Port mapping to the application binding same port in same ip address.
: SCTP, TCP, UDP

4 (OSI 5,6,7 layers)
: is called Applications. It used SMTP, FTP, HTTP, DNS, SNMP, TELNET
 

Address in TCP/IP
1. Physical addresses
	- MAC address
	- 47:67:
	
2. Logical addresses
	- IP address

3. Port addresses
	- Port number
	- In one IP address

4. Specific addresses
	- Domain address

#### logical IP address
IPv4
: 32bit unsigned length
: Unique and Universal

IPv6
: 128bit unsigned length
: to solve the problems,IP depletion and increasing IoT devices.

Notation
: binding notation
: dotted clecimal

IP address
: Class full address
- There are 5 classes, A,B,C,D,E.
- A class first binary number is 0. First 1 byte is network address.last 3 bytes are host address. 
$$ 
network address :2^7 = 128 bit\\
host address :2^{24} = 16,777,216 bit
$$ 
- B class first binary number is 0. First 1 byte is network address.last 3 bytes are host address. 
$$ 
network address :2^{16} = 16,384 bit\\
host address :2^{24} = 65,536 bit
$$
- **Not used more.**

: Classless address
- An entity is granted a block depending on the size of the entity.
: Restriction
- The addresses in the block must be contignous.
- The number of addresses in a block must be a power of 2($2^n$).
- The first address must be evenly.

##### subneting
: network ip address split
##### Superneting
: network ip address merge