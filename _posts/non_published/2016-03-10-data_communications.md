---
layout: post
published: False
title: Data communications
categories: 
- Computer Science
tags:
- protocol
- network
---

# Computer Network
Computer network based on TCP/IP

## seven layers of the OSI model
1. Physical Layer
	- consist of hardware communication.
	- It like $$0101110101010001,$$ just binary data.
2. Data Link Layer
	- Flow control
3. Network Layer
	- Routing function to send packet via nodes.
	- IP protocol
4. Transport Layer
	- TCP Protocol
5. Session Layer
	- ,
6. Presentation Layer
	- ,
7. Application Layer
	- data made by user to send other devices using network.

## TCP/IP Model
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

<!--more-->

# Network Layer
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

## Logical Address mapping
### IPv4
- 32bit unsigned length
- Unique and Universal
- The address space of IPv4 is $$2^{32}$$

#### Classful addressing
- There are 5 classes, A,B,C,D,E.
- A class first binary number is 0. First 1 byte is network address.
- last 3 bytes are host address.

|	 Class	|	Range |	Number of Blocks |	Block Size | Application	|
|:-------:|:-----:|:------:|:----:|:-----:|
| A	|	0xxxxxxx.x.x.x |	$$2^{6}$$(128)|$$2^{24}$$(16,777,216)| Unicast|
| B	|	10xxxxxx.x.x. |	$$2^{14}$$(16,384)|$$2^{16}$$(65,536)|Unicast|
| C	|	110xxxxx.x.x. |	$$2^{21}$$(2,097,152)| $$2^{8}$$(256)|Unicast|
| D	|	1110xxxx.x.x. |	1	|$$2^{28}$$(268,435,456)|Multicast|
| E	|	1111xxxx.x.x. |	1	|$$2^{28}$$(268,435,456)|Reserved|

> **Classful addressing, which is almost obsolete, is replaced with classless addressing.**

```x.y.z.t/n```

- netid(Network address)
	- d
- hostid(Host address)
	- d
- mask(Subnet mask)
	- f


##### Classless addressing

#### Example
network address : $$2^7 = 128 bit$$  
host address : $$2^{24} = 16,777,216 bit$$

- B class first binary number is 0. First 1 byte is network address.last 3 bytes are host address.

network address : $$2^{16} = 16,384 bit$$  
host address : $$2^{24} = 65,536 bit$$

- **Not used more.**

: Classless address
- An entity is granted a block depending on the size of the entity.
- The addresses in the block must be contignous.
- The number of addresses in a block must be a power of 2($2^n$).
- The first address must be evenly.

subneting
: network IP address split

Superneting
: network IP address merge

###c IPv6
- 128bit unsigned length
- New address to solve some problems, IP depletion and increasing IoT devices.

Notation
- binding notation
- dotted

----------
## Internet Protocol
- **Switching at the network layer in Internet uses the datagram approach to packet switching**[^1]
- **Communication at the network layer in the Internet is connectionless**[^2]

### IPv4
- Internet protocol version 4 (IPv4) is the delivery mechanism used by the TCP/IP protocols.


#### ARP(Address Resolution Protocol)  
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

#### 오류 보고(Error Reporting)  
**신뢰성 없는 IP의 단점을 보완하기 위한 기능**
Type3: Destination unreachable
Type4: Source quench
Type11: Time exceeded
Type12: Parameter problems
Type5: Redirection


#### 목적지 도달 불가(Destination unreachable)  

| Type  | Description | Reason |
| :----: | :-----------------: | :-------------------: |
| 0     | Network unreachable | Routing table의 entry가 없음(라우터에서 버려진 경우) |
| 1     | Host unreachable    | Destination ARP 무응답(호스트에서 버려진 경우) |
| 2     | Protocol unreachable | 상위 Protocol을 지원하지 않음.(layer3 -> layer4로 못올라간 경우) |
| 3     | Port unreachable | 해당 TCP |
| 4     | Fragmentation needed & D bit set | IP packet을 Fragmentation해야하나 D bit가 설정되어 있음(D bit는 Do not Fragmentation 필드 값 참이면, Fragmentation이 금지) |

#### Time Exceed
**TTL(Time to live)값 이 0이 되어버리는 경우**  
1. 데이터 송신 중 Loop 또는 원형으로 돌다가, TTL 값이 0이 되는 경우 발생. TTL은 라우터를 거쳐갈 때마다 1씩 감소한다.
2. 메시지를 구성하는 Fragment가 정해진 시간 내에 목적지에 도달하지 못한 경우 발생.

#### Parameter problem

#### Redirection
- Host가 전송하는 Router를 바꿔줌.
- Type 5
- 라우팅 테이블 갱신을 위한 에러.

#### Echo Request & Reply
- 두 호스트 간의 응답 가능 여부를 테스트 하기 위해 사용.


| Type  | Description | Reason |
| :----: | :-----------------: | :-------------------: |
| 8, 0     | Echo request and Reply | Routing table의 entry가 없음(라우터에서 버려진 경우) |
| 13, 14     | Timestamp request and Reply    | Destination ARP 무응답(호스트에서 버려진 경우) |
| 17, 18     | Address-mask request and Reply | 상위 Protocol을 지원하지 않음.(layer3 -> layer4로 못올라간 경우) |
| 10, 9     | Router solicitation and advertisement | 해당 TCP |


#### Address-mask request and Reply  
**Host가 Router에 subnet mask를 요청**
- Type 17(request), Type18(Reply)
- Code: 0

#### Router solicitation and advertisement  
**Router가 subnet내의 Router주소를 광고**
- Type10


#### Traceroute
	경로 추적을 위해 사용됨

1. Send UDP Packet with TTL = 1
	- R1의 IP주소와 소요 시간을 IP Header를 통해 알 수 있음.
2. Send UDP Packet with TTL = 2
	- R2의 IP주소와 소요 시간을 IP Header를 통해 알 수 있음.
3. Send UDP Packet with TTL = 3

#### Routing table
- **How to write routing table in router**

#### Delivery

#### Forwarding
	포워딩(Forwarding)은 패킷을 목적지로 가는 경로에 위치시키는 것.

1. Next-hop Method
	- 완전 경로가 아닌 다음 홉의 주소만을 테이블에 기록.
2. Host-specific Method


### Routing Protocol
#### Optimization
Routing is the method for router how to find next route in network.

autonomous System

Intra domain router.

Routing Protocol

#### Distance vector routing tables
1. Initialize
2. Sharing
3. Update table

##### disadvantage
- instability


#### RIP()
Using Hop count. Destination is not host,

## Delivery, Forwarding and Routing
### Delivery
**Network layer manages packet processing under the physical networks**
There are two methods to delivery packet in network layer.

- Direcct delivery
- Indirect delivery

#### Direct Delivery
**Direct delivery occurs when destination of the pakcet exists in the same network.**

- Host to Host
- Router to Host

#### Indriect Delivery
**Inirect delivery occurs when destination of the host not exists in the same network.**

- Host to Router

> Always, Final delivery is direct delivery.

### Forwarding
Forwarding means that 

- Next-hop method
- Network-specific method

### Routing

Link State Routing
1. 각 노드에 연결된 link의 상태를 링크 상태 패킷에 저장.
2. LSP를 flooding을 통해 전 노드에 알림.
3. 각 노드에서 전체 AS의 구조를 알아내서 shortest path tree 형성
4. Shared path tree를 바탕으로 routing table 형

Creation of LSP
- node identify, list of links(metric)
- sequence number, age

> ex) Node A
> list of links :(B,5), (C,2), (D,3)
> seq no : 1
> age: craetion time.

Flooding of LSP
> LSP를 생성한 후 모든 Interface를 통해 전송
> LSP를 받은 node는 seq no를 비교하여 오래된 LSP면 버리고 새로운 LSP면
> 자신의 정보를 update하고, 모든 interface를 내보냄.

Formation of sortest path tree
- Dijkstra Algorithm
This algorithm always needs two lists, permanent list, tentative list.
permanent list is selected as path.
tentative list is spare set.

#### OSPF

Formation
1. Store link state
[^1]: Internet selects datagram to send the packet to destination using path. Thus it uses all defined address.
[^2]: ddd

