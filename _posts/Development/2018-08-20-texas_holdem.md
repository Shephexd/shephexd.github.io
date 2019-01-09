---
layout: post
title: 뱅크샐러드 텍사스 홀덤
published: True
categories:
- Development
tags:
- Development
- Mathematics
- Python
typora-root-url: /Users/shephexd/Documents/github/pages/
---



## 게임 소개

규칙은 텍사스 홀덤과 유사합니다.

사용자는 2장의 개인 카드와 5장의 커뮤니티 카드를 이용하여 카드를 조합하고, 족보에 나온조합(로얄스트레이트, 스트레이트, 풀하우스, 트리플 등등) 중 가장 점수가 큰 조합을 가진 사용자가 그 라운드의 승자가 됩니다.



- 배팅은 라운드별 참여비: 1코인
- 카드를 두장 받았을 때
- 커뮤니티 카드 3장 오픈 후
- 커뮤니티 카드 한장 추가 오픈 후
- 마지막 커뮤니티 카드 카드 오픈 후



- 뱅크샐러드 홀덤은 기존의 텍사스 홀덤에서 몇 가지 규칙을 변경하여 구현되었습니다.
- 5명의 플레이어가 하나의 게임에 참여하여 플레이하게 되며, 2명의 플레이어가 남거나 100라운드가 끝났을 경우 한 게임이 끝나게 됩니다.
  - 두 명의 플레이어가 살아 남을 경우, 두 명이 모두 다음 라운드에 진출합니다.
  - 100 라운드가 끝나 세 명 이상의 플레이어가 살아 남을 경우, 칩을 가장 많이 소지한 플레이어 둘이 다음 라운드에 진출합니다.



게임 소스코드 및 자세한 정보 링크: <https://github.com/Rainist/pycon-2018-banksalad-holdem>



<!--more-->



## 알고리즘 구현



1. 항상 배팅

기본제공되는 알고리즘으로 항상 배팅 하는 알고리즘입니다.



최소배팅과 최대배팅 중 최소값을 배팅합니다. 현재 칩이 없는경우 0(DIE)를 반환합니다.



```
def always_bet(
    my_chips: int,
    my_cards: List[Card],
    bet_players: List[Other],
    betting_players: List[Other],
    community_cards: List[Card],
    min_bet_amt: int,
    max_bet_amt: int,
    total_bet_amt: int
) -> int:
    if my_chips >= min_bet_amt:
        return min(max_bet_amt, min_bet_amt)
    else:
        return 0
```





\2. 카드 EHS 알고리즘을 이용한 배팅

제가 제출한 알고리즘이였는데, 서버에서 시간이 1초내에 나왔지만, 노트북에서 1초내에 나오도록 최적화를 하지 못하여… 이벤트를 참석하지 않았습니다.



알고리즘 구현 전 고려한 사항은 다음과 같습니다.



- 각 커뮤니티 카드가 오픈되는 시점을 phase로 정의한다.
  phase 1: 나의 개인카드 2장 획득
  phase 2: 커뮤니티 카드 3장 오픈
  phase 3: 커뮤니티 카드 1장 추가 오픈
  phase 4: 커뮤니티 카드 1장 추가 오픈
- 현재 나의 패의 점수(확률)는 내가 가진 패 + 커뮤니티 카드의 패를 이용하여, 
  (다른 사용자와 비기거나 이긴 경우)/사용자와 지거나 비길 경우의 비율
  이는 다른 사용자가 가질 수 있는 두가지 개인 카드의 조합을 이용하여 계산이 가능하다.
  총 52장의 카드 중 내가 가진 2장의 카드와 커뮤니티 카드를 제외한 45장의 카드의 조합으로 각 조합별 현재 내가 가진 카드를 이긴 경우, 비긴 경우, 진 경우를 계산.
- 현재 페이즈가 마지막 페이즈가 아닐 경우(커뮤니티 카드가 5장 오픈되어 있지 않은 경우), 현재 나의 패의 점수(확률)에 대한 기대 값은 앞으로 열릴 카드에 따라 달라질 수 있다.
  연산량에 따라 조절 ex) phase 2 이전에 계산할 경우 연산량이 많아지므로, phase3 시점에 계산, (phase4의 경우 추가 오픈 카드가 없어서 카드의 잠재 점수 값은 0)
- 나의 카드가 이길 확률 = (내 패의 현재 점수) * (1 - 이후 카드를 고려한 나의 패배 점수) + (1 - 내 패의 현재 점수) * (이후 카드를 고려한 나의 승리 점수)
  EHS(Effective Hand Strength) 알고리즘 참고(<https://en.wikipedia.org/wiki/Poker_Effective_Hand_Strength_(EHS)_algorithm>)
- 배팅비율 설정
  캘리 공식을 기반으로한 배팅은 (배당 * 승리확률 - 패배확률) / 배당을 기븐으로 각 phase별 가중치를 변화하여 배팅 



```
import itertools
from typing import List
from .player import Other
from .core.cards import Card, Rank, Suit
from .core.madehands import evaluate
import time


phase = {
    0: 2,
    1: 2,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


def card_score(card_comb: int, card_rank: int) -> int:
    return card_comb * 100 + card_rank


def possible_cards(my_cards: List[Card], community_cards: List[Card]):
    possible_cards = list()
    for r, s in itertools.product(Rank, Suit):
        for my_card in my_cards:
            if my_card.rank == r and my_card.suit:
                continue

        for com_card in community_cards:
            if com_card.rank == r and com_card.suit:
                continue

        possible_cards.append(Card(r, s))
    return possible_cards


def hand_strength(my_cards: List[Card], community_cards: List[Card], phase: int) -> tuple:
    ahead, tied, behind = 0, 0, 0
    ahead_idx, tied_idx, behind_idx = 0, 1, 2

    hp = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    hp_total = [0, 0, 0]

    my_card_comb, my_card_rank = evaluate(player_cards=my_cards, community_cards=community_cards)
    my_card_score = card_score(card_comb=my_card_comb, card_rank=my_card_rank)

    _possible_opp_cards = possible_cards(my_cards=my_cards, community_cards=community_cards)

    for opp_hand1, opp_hand2 in itertools.combinations(_possible_opp_cards, 2):
        opp_cards = [opp_hand1, opp_hand2]
        opp_card_comb, opp_card_rank = evaluate(player_cards=opp_cards, community_cards=community_cards)
        opp_card_score = card_score(card_comb=opp_card_comb, card_rank=opp_card_rank)

        if my_card_score > opp_card_score:
            ahead += 1
            cur_idx = ahead_idx
        elif my_card_score == opp_card_score:
            tied += 1
            cur_idx = tied_idx
        else:
            behind += 1
            cur_idx = behind_idx

        if phase == 4:
            _possbile_addition_cards = possible_cards(my_cards=my_cards + opp_cards, community_cards=community_cards)

            for river_card in _possbile_addition_cards:
                our_best = evaluate(player_cards=my_cards, community_cards=community_cards + [river_card])
                opp_best = evaluate(player_cards=opp_cards, community_cards=community_cards + [river_card])

                if our_best > opp_best: hp[cur_idx][ahead_idx] += 1
                else: hp[cur_idx][behind_idx]+=1
        elif phase == 5:
            continue

    hp_total[ahead_idx] = hp[0][ahead_idx] + hp[1][ahead_idx] + hp[2][ahead_idx]
    hp_total[tied_idx] = hp[0][tied_idx] + hp[1][tied_idx] + hp[2][tied_idx]
    hp_total[behind_idx] = hp[0][behind_idx] + hp[1][behind_idx] + hp[2][behind_idx]

    hs = (ahead + tied/2)/(ahead+tied+behind)

    ppot = (hp[behind_idx][ahead_idx] + hp[behind_idx][tied_idx]/2 + hp[tied_idx][ahead_idx]/2)/(hp_total[behind_idx] + hp_total[tied_idx] + 0.0001)
    npot = (hp[ahead_idx][behind_idx] + hp[tied_idx][behind_idx]/2 + hp[ahead_idx][tied_idx]/2)/(hp_total[ahead_idx] + hp_total[tied_idx] + 0.0001)

    return (hs, ppot, npot)

def bet(
    my_chips: int,
    my_cards: List[Card],
    bet_players: List[Other],
    betting_players: List[Other],
    community_cards: List[Card],
    min_bet_amt: int,
    max_bet_amt: int,
    total_bet_amt: int
) -> int:

    current_phase = phase[len(community_cards)]
    phase_weight = [0.01, 0.01, 0.03, 0.05, 0.05, 0.1]
    current_bet = sum([other.bet_amt for other in bet_players])
    other_min_bet = min([other.bet_amt for other in bet_players])

    # EHS = HS * (1 - NPOT) + (1 - HS) * PPOT

    hs, ppot, npot = hand_strength(my_cards=my_cards, community_cards=community_cards, phase=current_phase)
    ehs = hs * (1 - npot) + (1 - hs) * ppot

    dividend = total_bet_amt/current_bet
    win_prob = ehs
    lose_prob = (1 - ehs)

    alpha = 0.20 # risk tolerance for loss
    my_bet_prob = (dividend * win_prob - lose_prob)/dividend
    if lose_prob + alpha > win_prob or my_bet_prob < 0:
        if total_bet_amt < my_chips * phase_weight[current_phase] and lose_prob > 0.4:
            return min_bet_amt
        return 0
    else:
        my_bet = int((my_chips * win_prob - other_min_bet) * my_bet_prob * phase_weight[current_phase])
        return max(my_bet, min_bet_amt)
```





3. 카드 스코어 기법을 이용한 배팅

대회 이후에 구현한 알고리즘으로 노트북에서 1초내에 동작합니다.

배팅 로직은 (내 카드와 커뮤니티 카드의 조합) - 커뮤니티 카드의 조합 > cutoff 인경우 에 따라 배팅하고, 그 차액과 phase별 weight만 계산합니다.

커뮤니티 카드에서 원페어가 나온경우, 내 카드에서 조합에 기여할만한 카드가 없는 이상 점수가 cutoff이상으로 나오지 않습니다. 

예를 들어, 내 카드와 커뮤니티 카드 조합에서 원페어가 나오고, 커뮤니티 카드에서 원페이거 나온 경우에는 배팅 금액을 최소로하거나 배팅을 하지 않습니다.



```
import itertools
from collections import defaultdict
from typing import List
from .player import Other
from .core.cards import Card, Rank, Suit
from .core.madehands import evaluate
import random


PHASE = {
    0: 2,
    1: 2,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


def card_score(card_comb: int, card_rank: int) -> int:
    return card_comb * 100 + card_rank


def my_card_strength(my_cards: List[Card], community_cards: List[Card]) -> tuple:
    my_card_comb, my_card_rank = evaluate(player_cards=my_cards, community_cards=community_cards)
    my_card_score = card_score(card_comb=my_card_comb, card_rank=my_card_rank)
    com_card_comb, com_card_rank = evaluate(player_cards=[], community_cards=community_cards)
    com_card_score = card_score(card_comb=com_card_comb, card_rank=com_card_rank)

    if my_card_score > com_card_score:
        return (my_card_score, my_card_score - com_card_score)
    else:
        return (0, com_card_score)


def bet(
    my_chips: int,
    my_cards: List[Card],
    bet_players: List[Other],
    betting_players: List[Other],
    community_cards: List[Card],
    min_bet_amt: int,
    max_bet_amt: int,
    total_bet_amt: int
) -> int:
    phase_weight = {
        3: 0.02,
        4: 0.02,
        5: 0.03,
    }
    current_phase = PHASE[len(community_cards)]

    if current_phase > 2:
        my_strength, diff = my_card_strength(my_cards=my_cards, community_cards=community_cards)
        if my_strength > 10:
            my_bet_candi = phase_weight[current_phase] * my_strength / 10
            my_bet_basic = min_bet_amt * phase_weight[current_phase] * 100
            my_bet = int(max(my_bet_basic, my_bet_candi))
            #print(current_phase, my_bet, my_bet_candi)
            return my_bet
        elif my_strength < 20 and current_phase > 3:
            return 0
        else:
            return min(min_bet_amt, 1)

    return min(min_bet_amt, 1)
```



P.S. 구현은 해두고 이벤트를 참석하지 않았습니다. 서버에서 1초 내에 구현 되는 2번째 로직으로 제출을 했는데, 노트북에서는 시간이 더걸리더군요…