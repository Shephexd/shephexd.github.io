---
layout: post
published: True
title: Mips64 analysis
---

###MIPS 64바이너리 분석에 앞에 MIPS의 특징에 대하여 서술하려 한다.
>밉스는 연산을 진행할 때, 64비트를 한번에 계산하는데 제약이 있다. 계산은 16비트 단위로 계산하는데,

t9은 함수에서 호출받을 때의 주소, 즉 시작한 함수의 첫 주소를 나타낸다.

```
lui	$gp, 0xFC0 //$gp의 16비트 앞부분 주소에 0xFC0값을 넣는다.
addiu	$gp, 0x76E0 // $gp의 16비트 앞부분 값에 0x76E0값을 더한다.
addu	$gp, t9

which IDA will simplify as:

li	$gp, 0xFC076E0 //$gp의 주소에 0xFC076E0값을 넣는다.
addu	$gp, $t9
```
##0. 용어 및 개념 정리
###MIPS에서는 파이프라인


##1. GP(Global Pointer) 찾기
###Mips의 분석에 앞서 가장 중요한 포인터는 GP이다. 종종 IDA에서 분석을 진행할 때, GP값이 오류나는 경우가 있는데, 이를 계산하는 방법에 대하여 정리하고자 한다.

####1. GOT값 기준으로 찾기
OFFSET_GP_GOT값은 0x7FF0이고, GOT의 주소 값은 아이다의 특정 위치값을 조회하면 알 수 있다.
실제 GP값은 GOT address + OFFSET_GP_GOT이므로, GOT의 시작 주소에서, 0x7FF0값을 더하면 알수 있다.  
This value is always OFFSET_GP_GOT bytes after the program’s GOT (global offset table), OFFSET_GP_GOT is always 0x7ff0.

```
.got:000000012023CB10  # ===========================================================================
.got:000000012023CB10
.got:000000012023CB10  # Segment type: Pure data
.got:000000012023CB10                 .data # .got
.got:000000012023CB10                 .dword 0
.got:000000012023CB18                 .dword 0x8000000000000000
```
>hex(0x12023CB10+0x7FF0)  
>**0x120244b00L**
####2. 함수 앞부분의 계산을 통해 찾기
GP는 함수 내에서도 위치를 참조하기 위하여 사용한다. 그렇기 때문에 함수 앞부분에서 GP값의 계산은 필연적이다.
```
.text:00000001200ACCF8 var_28          = -0x28
.text:00000001200ACCF8 var_20          = -0x20
.text:00000001200ACCF8 var_18          = -0x18
.text:00000001200ACCF8 var_10          = -0x10
.text:00000001200ACCF8 var_8           = -8
.text:00000001200ACCF8
.text:00000001200ACCF8                 daddiu  $sp, -0x30       # Doubleword Add Immediate Unsigned
.text:00000001200ACCFC                 sltiu   $v0, $a0, 0x10   # Set on Less Than Immediate Unsigned
.text:00000001200ACD00                 sd      $gp, 0x30+var_10($sp)  # Store Doubleword
.text:00000001200ACD04                 lui     $gp, 0x19        # Load Upper Immediate
.text:00000001200ACD08                 sd      $s2, 0x30+var_18($sp)  # Store Doubleword
.text:00000001200ACD0C                 daddu   $gp, $t9         # Doubleword Add Unsigned
.text:00000001200ACD10                 sd      $s1, 0x30+var_20($sp)  # Store Doubleword
.text:00000001200ACD14                 daddiu  $gp, 0x7E08      # Doubleword Add Immediate Unsigned
.text:00000001200ACD18                 sd      $ra, 0x30+var_8($sp)  # Store Doubleword
.text:00000001200ACD1C                 move    $s1, $a2
.text:00000001200ACD20                 sd      $s0, 0x30+var_28($sp)  # Store Doubleword
.text:00000001200ACD24                 move    $s2, $a3
.text:00000001200ACD28                 bnez    $v0, loc_1200ACD50  # Branch on Not Zero
.text:00000001200ACD2C                 li      $v1, 0xC         # Load Immediate
```
