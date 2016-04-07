---
layout: post
published: True
title: Data communications
excerpt: Introduction Data communications.
categories: Computer Science
tags:
- protocol
- network
---

#Computer Network
##Basic definition
Routing
: When use network, It is route setting to send specific packet to other places.


##Network model  
**network model**  

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
	-
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
: network IP address split
##### Superneting
: network IP address merge

###Network Layer  
제 3
####ARP(Address Resolution Protocol)  
**인터넷 주소 매핑을 위한 프로토콜**
**아이피 주소를 물리주소에 IP Address -> Physical Address mapping**

Proxy ARP

RARP(Reverse ARP)
**ARP와 반대로 Physical Address -> IP Address mapping**

ICMP(Internet Control Message Protocol)
**IP를 지원하기 위한 프로토콜. 대표적으로 ping message를 활용**

1. Error Control
2. Assistance

- Message type
	- Error Message
	- Query Message

Layer 3 protocol using IP

Message format
32bit
Type(8bits) Code(8bits) checksum(16bits)

IP header|ICMP header|IP header|

####오류 보고(Error Reporting)  
**신뢰성 없는 IP의 단점을 보완하기 위한 기능**
Type3: Destination unreachable
Type4: Source quench
Type11: Time exceeded
Type12: Parameter problems
Type5: Redirection


| First Header  | Second Header | Third Header         |
| :------------ | :-----------: | -------------------: |
| First row     | Data          | Very long data entry |
| Second row    | **Cell**      | *Cell*               |
| Third row     | Cell that spans across two columns  |dd|
[Table caption, works as a reference][section-mmd-tables-table1]


#### 목적지 도달 불가(Destination unreachable)  
	-ssssss

| Type  | Description | Reason |
| :----: | :-----------------: | :-------------------: |
| 0     | Network unreachable | Routing table의 entry가 없음(라우터에서 버려진 경우) |
| 1     | Host unreachable    | Destination ARP 무응답(호스트에서 버려진 경우) |
| 2     | Protocol unreachable | 상위 Protocol을 지원하지 않음.(layer3 -> layer4로 못올라간 경우) |
| 3     | Port unreachable | 해당 TCP |
| 4     | Fragmentation needed & Dbit set | IP packet을 Fragmentation해야하나 D bit가 설정되어 있음(D bit는 Do not Fragmentation 필드 값 참이면, Fragmentation이 금지) |

#### Time Exceed  
**TTL(Time to live)값 이 0이 되어버리는 경우**  
1. 데이터 송신 중 Loop 또는 원형으로 돌다가, TTL 값이 0이 되는 경우 발생. TTL은 라우터를 거쳐갈 때마다 1씩 감소한다.
2. 메시지를 구성하는 Fragment가 정해진 시간 내에 목적지에 도달하지 못한 경우 발생.

#### Parameter problem

#### Redirection
- Host가 전송하는 Router를 바꿔줌.
- Type 5
- 라우팅 테이블 갱신을 위한 에러.

###Echo Request & Reply
- 두 호스트 간의 응답 가능 여부를 테스트 하기 위해 사용.


| Type  | Description | Reason |
| :----: | :-----------------: | :-------------------: |
| 8, 0     | Echo request and Reply | Routing table의 entry가 없음(라우터에서 버려진 경우) |
| 13, 14     | Timestamp request and Reply    | Destination ARP 무응답(호스트에서 버려진 경우) |
| 17, 18     | Address-mask request and Reply | 상위 Protocol을 지원하지 않음.(layer3 -> layer4로 못올라간 경우) |
| 10, 9     | Router solicitation and advertisement | 해당 TCP |


####Address-mask request and Reply
**Host가 Router에 subnet mask를 요청**
- Type 17(request), Type18(Reply)
- Code: 0

####Router solicitation and advertisement
**Router가 subnet내의 Router주소를 광고**
- Type10
