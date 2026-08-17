"""Private reference support for the tensor-backprop exercise notebooks.

The notebooks expose their inputs, forward operations, outputs, intermediates,
and upstream gradients directly. This generated support module retains the
reference sources used by the private test helpers; expected gradient values
remain hidden from student lesson content.
"""

from __future__ import annotations

import ast
import base64
from functools import lru_cache
import hashlib
import json
import zlib

_PAYLOAD_SHA256 = "400938f946064d0854354486bc28f1832bfdb58068546b0ee5681523aa89e38f"
_PAYLOAD = (
    "c-"
    "rk<YjfK+v;Hfb^o#9Ss(91Iot$aX%bU#U>DxK&)NvG#hLUJYClaZblwvuZ{`Z5$g#bu`1SL|Jw)qf?BnW`}>@Ic}3*e6zosRk9=E"
    "YaWubb6sG54p&!+g8mME;EtZ31IGo5#l3e+VPrc#7uhdGLtt`Nq-<Hr~SUg80dg+8=|h@yb|-(PY+M`#~H=7goD-ZA{nCEC1E^zu"
    "o=v!{x_d6r&RU2-"
    "WrEMbnFqFZ?ZP{NsyjV{5!L>~;s0`O0_~#4CTYHo{=>jGKyC$L7Imvo_+HxANm}j7hjemp(2M`TyDY@fwv0H)|;L9B!i6Up%0mt)"
    "h9bMvcA0Z_|%MZ+O$^_Qw~f$rsm%238X^7@9zY7AA!a|2gf>y%tq0LX|7SYP(0IirpMliiz@^d8vip>a<<{svaqA#$eA!B&5>_Au"
    "%BcVf2WI>FqM6)T5unSn-"
    "(jb8Dra7kR;>f6<k{za9Cw^A{H4E)_Lg#!Y7DsQRkg1VPMeqgPCmEK3ckTvAR%JT;U>TaU}Sr@&>s;}A8MAU#ZwcDz|$Tpp1m{Sz"
    "WdpOa)dU%u)blP-"
    "f3qRT+0i+M!4n4Qz1i`hxh<=Xh<`z!eQTeR_yNhPqjbGvM&SUlTh{JwP}!=Xv+)TF45jI(3X)oz6QtOCNo2P<e}%d|nL!rvhbk*%"
    "Z3XdTAr!hcw66ADquL5)ZA$64m3zTzFHfU?*m#Xb1ez?B}6m~~jDYj3>pGt_hylyQkX&ZlwK7()nP6AV$$<dzpHv<QPo6-"
    "BXGi@PyI_1+(mp4~<~CE|l)?=j;zUgKWR5qU?CkvAopmYBL==p!5UV=zUZlc}`RwwU{s>y1vl74)5w7+bi{*Tx?oUre_jU)&hmKh"
    "Ybg-b+rJErwyUS|sY?^4j?L;+fxm7H-4K{0Usqhm4lJrh^)opiTvXwh*C^k-t??lt%`byKlK=bnP9UuKlLHdAABlw@UV8v67Ysd-"
    "gTR_#@F7=p&}aW)-iI>@KMqxXB}>l7NE-g}jBo+pH|=S65aMR~BiIEHwuQq^8w5DV2G3Wvlh2%Q)OBrJ>b1Bn=yhfeRw&R3l-"
    "_EaXK^Hnhw`lF(|1gccU(9cYNCRHPaSgCj*zMn&t8RBR^X-r`ZZ%m)`yvF&eH7nceuN&|H%741V((P>zms|?UvWO+X-"
    "iO3BdEs1sxNkzBuVVY620x{QKFjG=9Ye1IrrF%$<cAJuJmr8dvTE_FoWjLSeNqDI#Zgmey%4VFKmMz+lRR<m9qc1i(%xP&dXYY^{"
    "?KeN$t3}E$MYgC@(S83?XaeZkqFL_<JBnZ6=a(!buNMo023QQzU~J61&pv@A3kpM^8NbLT!+!3puQmkaeJ7W13n6GJA&}g)QV0eI"
    "fxzxG?c47z1x~^2eIW!_B~qYi2-}^*KwvfqfnCLV`%;%fXMC+hKyc{F^^k2I27yy^<t(KqSIO4159!I-_5q@=T11Q|o-"
    "n;2QO;&sU9Y0q{6V2ZOd$b8UoDZIy)<yewFrovFCe%fxWqrmE!-yDf*<*^!(?Vo5%U-"
    "R(ht^8^Vm1g_pD}G(&QW`x3JjqhrE(}id${BgvidpL+-ZQ+?e20ADamcm%bOMh}g=@f0e1|9-"
    "fNbhAsSRRDAMb?lvwmN}Q?@s|!#owBpMmE;sRDR0vQvUzgC@;RgMg#SPDK#`)lpnNhX6I7${;TcYUbLPt>`rQ9o{VZF0+j70sDB2"
    "keZuuIiIzYz)zPL7F4%AVE~TiGB#IJo9?Iwwg_vovs5LQ1D|cv70DNy=2Jor17KVuRwa{e2m6&BGJdI%o!*Q?3{56MM%xJUI`WP>"
    "A#7%E_tAsx%#+28+O~D&<)F@FaE)kPA}Lw7-"
    "CmLy7fRV)!^t6T`aGj2kp$5x|5_>G1e2q|+FnG=%=}UXoL<m$aA%@yeU{7tB{jgpJq!Dt@(-"
    "TBQWm)va`9uypRuIedDe(`zu*!A^YAO0yO1rMFebMM_M0rx9nzX=Yk(XeRzOB@>o=TS`J?A-7>K(GPf*xm!gg5TX)o(EjW#Hhx@-"
    "InX~m$DWE;H3=7+We{h4D<C(0RveuwAHPH&i&hCVrav*YDiS@Zg*{|F)sX_Xa}t)=c<#j|Q_0mc%%Rd0PJGZXiR4+Og9cq>A;ah@"
    "rtEN&=nDR714v$CPvA-qI`ed)8xlP;6qc^upA2^931D~lm1a56<<2{$Ragm7MN>@KfrsTxP{A(qn_NpV?Nl<yqyUS)#vb8IJ-"
    "F<K1}M|6C~f(3j{9j!I$x+1H|p&5b`&?QsY87X$!iIQaH)r(bJ`g0sWKY|(K!(i-"
    "MZIUrlUGmVz28_UtWAEf+<O)Qw2@j!IxwSW>bTp=>J1|03!Uwu#%S19sDB;D|vxo1Xluj^kw!GU+RI|Z5mv4$nMbSUNwtu_XN>xW"
    "d7B13Kx-Auvl~TZ|yd#%>XU!cm~tn37D<@W<z)#ZR(tcX~bfId!X7sk%4va*NX$JB43hBfE3*;o_{FIyl}+;4yp|>LVtw7O+!+N+"
    "hWh~XK29M?ff+z7Y1!2pA6gma>u$IEpVF=&fNeuPjylSkVs8eXXaPgN(LxWreiI=o@5{uqdmACgd6<bCDcmZXHVct4<zfvT~FEfa"
    "jJQqUzd2E`<b#`>of+)uHE2tQ99dr$mfk6Ve;4(r;o_u9z8E6d>WxauOu&rp-"
    "D?Og{K$}VoiZBLVoTdl}Rm$Q$lX&;p&{o+|z@%Sd6{Nr_xbN0rJv|KV{hq9$p(idW#tSfeYXxdMVz$mo}k%n)wJXsPP)ajdks^sW"
    "@*}5nf^q!yWcpn=Sau>?ymW#~AJ@gy=O4Vl5Rw!y?=mW|0jGZydx8<bv2!eQ>+ScV)YE1e=s|UzKxw>d+wmcI5?A^ewM21>59W++"
    "sE<q24NLe{jXv6KfDqWxq2}7wY^%czN}#0K22d1HEG#&&TwGvl}z*U~}h>@x8P1k^2W(N2Z%E^^}jhu$rpcBIR$5ryX-GyNcM<6J"
    "DaDjo1^v{iFv-"
    "9kHm_b+}90TEmI25f<AzQnb2a4C*zY@w!llzUHKUEg(=iWv<sbm8G}0ic#0(s|b=DyJY8)`$|%CTe1YC45XeSMd%iO-fBzFEx&!M"
    "2a?&4^i(xR-l%*>i^t0=4~%zIj^3z-"
    "E<r#^6Mc2Jvm;jzM~m(<lJy3DzR{MfH~jV+;r1QBeU}2QjtJ5FCc=0PN*ZAKK1b40W(ZjgSqvuI)mR4862N>92)-|r!1o-"
    "U_vI2;0A?Mbq7N!C&9)vjDqEG>n0+Xj>)aPMTY&8YVD&-6T>QY{_@HYp7C`9y4dqe69(Z$6K(m@qH_RkXk!)-"
    "5OEqENm5tnn*oYgTgmwe<F!~qaqD2_TqeVDDlzC;`?X`xihr>_2Lc&CMz$<r{b86(L9NvK2(72FdlXNA;KHY!Vry2IyZ<@H*Jr@S"
    "*{lh>_Fwn&vUo5E9LS;<UI~OME*UCgYPK01Y+P@V?t;K`VSUO)4rYZB+`x%R3U;37<d+QBOC7$x7wm=u)v>o1D8$XAEe-"
    "B$E4&m3eadY3$jB;-v0H?8epCZumh2Q?7W!&{UwFX*t2$G=t6@o7e(Y{W-w!($HOX?jjmlv<EjW?HN7QUkqiLvr*{C?*+!I-"
    "{zAm?wFEZ1AL{OlYT&=16DvrYy}^Ek1Xel3ADQ-kPL2ekKAwo+kpjom73I$CE^n+SDY9U&izdm;~$Gc`)a-"
    "@X*BGck*p#1OLZg;mTjrob_#z#H!^;eoXmFWCpOruPU6WfpvO%2>%RR|-{p?-"
    "HxGZ<koTDtz}cVf)TM$d$Rh<<L1tfv(XI+8+A**35;j50yGQ&u_hD*qNvWE6<4Q=}m7HjrDJ#0jpl^?F!oaEil5{j1!A>`YqAvw^"
    "*m&TG}K3TYmekp0+-"
    "j6AwL9Npe0POfP$A1htgMA67~Q=!@A_9?;a=C56`Gw>_;{=6>(Awik>4v+*LF`bG|%12WTld&P?TVk#4@5NDl}pS#dHsU=B5XZ_T"
    "nd;e6<vGTX8(N;XmiY6G2_TAgI-"
    "<J~zM_ZabINI()Zd!woAUM*?cIgjJ<ghE%e$FL(^e@6k3J^Uf3vY8q?$))Jb6?Cv!d2sy#_(eotY{6~CV14uYS1~YBehhJ=^S}El"
    "aAxUpv^P^E8l-=AZ8`a0G-(`K+X8=nI1sq>7M^0<X2kSDYfBrGg<RfVEFh$A5|HJ%5~zvEFXqw!z9@j|4b*`sv@Gvn>=+E-"
    ">7TQ8d#@dijiI?Rsu4seJ@IzXSC2ugFLRoOdD7k{5X6V8`>6VPyDMmp;I`2W*m9JWLD)EnnAN?VY<~<r7#nMElS}woW^pLYKQ3!Z"
    "2he<n#904t@GHg>d8zVG_pPK!2MgU(-6lLvRXq5=P(VtS5Rk=n+u~HVQO8<QLtJ=FE%vjp31uFD13@DCkSx-"
    ";Bgg$fLNS`xOK}70%(;kt&MkFR~t)<YV}su4SGk8*s0>8x9WhMt(7gVuK;MaiEx0ITR{cLnI|*4wMFg$y>%_w8sIp`EO*IPjfp4)"
    "BA??@Fovo(0fq0*F1XG^lO2vkqMlZ6*t5!Nlpce_YMaoCjVIvnbU5a>PP<DR+$XKcG%=&uJ*cp6<2KE~-"
    "n_=n+m!~K*`7X3KX0fJe9#)$t{Q(;7-<Gghux^M*fI9D4%ZkWDgH**^P>kv2c?(Ccm`sO2iFJm$3HM4`-kXgrR)*uv&K+kb~-"
    "1u)&0`j&X;tb4Z7hy-eZ=;rD2~NQb5bn);P8veLs+fwn4YJp-"
    "sRpH{|C|8{M#4lD%%ojr3-^Qbd5<NFih%Xves!(|6SMl#e7y{LjRw*7HHjquPiNHTj*Y&}kE!HkzF}A~$Pqj5i&+kj-pYff#W9rS"
    "Y|4cJgprhoAglGzW=7_O!no0~XN0kUhDt!I4ED4m3g4Q?&;zV6_K7vY^oCi3Wp9aSmcR7G@xpWBnXavtu`hfXdkI79ud8W(^TAYS"
    "2pwQK__s!Q-V25_<W*aN=29jy9B*&*RVzb*_oLS*7#Rh<tV8h<eW-qKk5f8H&z1pa@P5$;CVZkbI+&{Is+zw9aWj>oy0irH0n31}"
    "$J~5@11CfzJ0kfxJW3wjw7KQSLMV>^26VWhzM0t1O#Cm!jt$7&<FwS4Bj#(}1q`SC_qqg$Dq_mkXG5FKdCeo5dA}fuK>EegWjaAw"
    "X;3Twv!wox)CR=qd$e|E~_^DhwCc6U2lmJIHURG(1E_pfPEwZUjlKLU;vVr%)XHtx)7ght>?^;J7fFoxiVMz1?T?{%*mxHMq$ti)"
    "^awT!SDJ5^#lhDbc$(RHq+CrSG!3!ZhoM6hX!qISMfcMYiC@OvXl6Jw@W?1|PiW(O>5}POw#PKJ&%?!k>^BAn)FgNP~@tLMyE$Y+"
    ">Rk6Kj#D<<a9IX>K05$~pK$C@JRPk$YmmmULOP=<p3OJV^T>#0Tk}6zo<#u+MW~8S*%;5LV^8TuZa`9k!)%;;vF{%_i~kqYW`?(j"
    "U=85DJ1tCOV)`{EW`je#`X4PE^HL#yiBJh%`nRES~ZA#f%#BV1-x$6D<D5n1oAo>Ej~6oPG>!g)oFd&*3IQBz|afJm2BBDMis6-"
    "t@Ws@dYEiT|ePd;aN*CL#v7!?BR@Pmv>OSMds^F5;Ab~=FfsKS`rC_4z~<if7x`ZqxS(BImBTnXfu()`I^Ze`o-"
    "qwaTn<B*XU=iNY2-"
    "?IL;+GODYjJzeYENBG3)~9IjF#SVwRQsvRlW!AT@+M|PNmn*i4=w8JL9nn4gpmq@b&L!o7WOt{R#CY2Q_Kgs||nh;NKh3gbTb;Ot"
    "w5@wbZnNnKCj@Zy0=2`cZ+srG?<l+OmvH-apAhL5jIZ~HU^M*djLQ_a@4v4&?JinI$rdt~@lztg)mmJAZ<=iq}<!By!PGEr>S63~"
    "wMija-3ecR01l!)9U#So&A-"
    "IeTG?P~t?xzsyorma9oHA)7NwA#!F)d9FSViGDPJlPAQW!O*HF5k1x1BGKc;HLpc$4bfUP7zE8YB}5^-"
    ";=rAE%XW9T6!8Lk3rz6J+7UAo$O8#hDQY-%3j0|4j0z<MQIJ@K7(`I(%duA|ITW7*ZQA(0rdqfgLE-Z%FgGD-"
    "l$v(Dgy766{V2x<Ng*Z9aOfWD8*QKXWQA>|^NVi4boImK~(ML)_qyJik}2@hQpJL7$)N1t!F2zjMvaVggrj?W|4{C=~efR!6-"
    "Ux{^O{6MtSg5t;`Yap;|@4!usAkJV`u&l<gXtBxR!@cwch%v)n`J(=AY56Hna{H?b{&Bv%T)c)BwCg|6E>LIsyvhd;<JzDumIaFJ"
    "maZ4Bw9|HsZ8{)|l5_e;%nndm0U(*sqyz$}5ST|dA>Uxlru&5d)`Ik5slV<bL0yx_f@9D%+KuOsil+atcv>X+LpKKcc%xBl-"
    "2##WlW$$CyK^zLy{(+qe5|fZ^96qeU^#&5-Q`~<3>8Ec9dhrB-hlr2Yz92Y~Kn24^AUJrTI1b0#EeVE@o=T1+N3Uhb730E*VJ5%G"
    "Jj0{}HLjEE0{=vMhgxOQJz_wkTTCzUeq0%uw;4Tz!qU)-pT_W{B^0FE1ilK%ABX-vNlX)U1ZFWLqy#s(AFsmzCS1uQ+6KM)gZv@y"
    "NP0x*z_nCJERpj#YtLoKpn4%wIIo~oNG_0a%4IgVPJd~_JITkgJ2^$AQmnwA=&zt(f2E?rNKS%mH-ehX@%UT+Wyo2_6rYqTwQ5nN"
    "--Q8sy77&NC|t_H#W>x07Af?`;b%~tpiZN8E@v6@z!?4S4?q4&J8K`Wvq(!JIKqh42TF79gOFGH0_W_%z&-mf=$`!-"
    "^v?bZ>hOAJtH>V*O>BQZ#^=h;VKoYFqv&*5t@F*wOm7^}psmssE4X32Yz6<yTHUNfvfhMrRoG6R)F<pHPi~|}fWetSQN#InV`RiX"
    "JS4CpYmd>Wkz;`on-"
    "g52)qGgVY9j{J4eL++Xp<t)5wQeZ%j{s+;Z7%XZh#%2X4tZf{Ho=6xqR#f96O7E$LM!MZ|9sNSVUnCft4;{r9d!i1Hn?nlP!(y<D"
    "7zRf2Dq7CvO~l#=Bf3PpI>gR++}v#>E>XdS*?^)9w(UKErkv943q7WT}8s;y|GK3|nRJ0_ZZm;qMy;m{kj4gn&8Gd45c@o}xprqK"
    "Ztv<<e@Yi%Gu1cxj*lm&Vl<PJyop1~Y+yPtQ;q-"
    "jMrwFci`_sV!9`I{{<cC$6?Q2HCkcimv|Q13~0HC6WdQ>n3z+Mi<%^!4@@}uL<cemY{XRyR?&%EQOsEL;Vqppvx3jIJIyEp>{T1&"
    "y}a;sx!r-HxKgU&Z4nB2=nQ(gzA)FlT94J;&83Hz_jr2wo|l+R|I-kza5mHwzNB!;dB<-I^#18@eSiWd<*(S?!hm5Gg+}6<geK-"
    "U>2x>)XAU1wYaRCbV@eAyh$p%+;;1YOFwKD3*!yMrGJLJB81T@H$uIhMZRy$=F_Pk7=gcu!XU<si^w}lndYgmbb33bB#@lUJ(<F7"
    "el#3jI(41IwpredO$uwWEOalxY{ok(85D54UMV3p)+Z~ZMsc|1tuV($`n&>9lD$U<Aql>gA=N4uwvy6<QOs^Iqf97XXqquth=C^5"
    "!c7QX@k=8ifM8Cr5^R(bKI_Uz)hMzlQ#u0dygXAt96>U#@Ee{nU?M}nG|(-"
    "OEf$<GyeOJ~28l~T9f7}H6Nxgyl^=QQc^D)zGv&S6SH+PE!}0(5lQm2vB?M$murFk7Cd<S#k#{^pg#*u-"
    "*n6s#+!(y5I5~x{Q)Rdw<%CX`Npy08OuB?*2uOJ6luVUMmk^A3frs--r3<CY2wt`zOL8ghpo{V+#O2{{<gXFwMworD<jC=QuCkR8"
    "`>e{NLHoQS(m?Z-@eAT*n2C(n-"
    "Xmi9$bax7KbWAw01Yn2%3IHX`@OZX^rJ`LSfH{nurPia(R~+3P*R7v68es&NdMtcP~<v^*7<1T$lox(k^3R%se3|4-"
    "qXrGMD5N!M2k!xIb-GYTBCoMPig<~Sl?6>Ifr?aGbVZcO+{Mm-Vyql1c`H_!#ch0pJlHH=h$nfbC$g}&$8E6y&eK8^NdZI)@0^IX"
    "q@mP<9~ko#XuGvGKA*D4_Gc%BM8|!^yg3i{_ej?S{PjZK=wF+pKvUMOMd>2es;-"
    "E*|`16_!0aMSR}IWVU<XPMndQ*^GLoy1JcU(AaCj2_c7}HCm)u9`O_(s@+Onba<jnZmJyB-"
    "<v!zO+N7!XW0jxMVT<sX%Ym1So5{6}p9`J}POLH|XGF?Yf#&YUiX<0Qm7D(>5a4_C-"
    "4Wyr#q0S5jkl4vggfZVgOJl>v%={@A^PAb1~$O(2I1Nthv6ss!H-"
    "D?{^bW?g`z%Owh7>)*!QOwmg)4IzTI_twA0`5#f}kT$l&4v<8^IdFp(AU$ZC3pTae9DWNvoaxr+y(Fg|T0TRGvFE7AgK=&e>mwyL"
    "3VrXgIZPMxN*PUa|rk&avi=`+kK-7aW?dX5(jRE`f~$rAr9sr+}X$}MHjtvrJNzH^6hMi-"
    "ZAq!?-V!Fza^2VitQV_k(62woS9d3Q*p{62Xw!3$*RMdCYQrX$}i@gDt{B%~o!@~uG`_@zM@ER}V})w(x{AQcKUD-NYFiEU90hIc"
    "Y;Gr*Y>&)Y3>;mJGHffu^kGKAX1ViqkYIPT<<xa~s0LZ_Xmc;d1r?=y#;S4e0hJL}{ran*(HUl~$8obbdg76gu!Bn~<k6Z(ok2=H"
    "Ur{-OZRzsZ_H%qy6Od_|5RMy!R#+bCQ;Q)rUpJ<<xr-"
    "(p>`gi=Y4Ab+nsNf64)S%N~VEhAMqDW4<?re0>5g1ih(734Pxas}nLD5~;Jk}U}LSO%cbohu_-LqZ*~yp%FXo|BwGu`Ep*#2j}?W"
    "b|%am@$Z3PrGca`qFekv28`ZVAA)pguyHwiZTY}(yEj}UaBx>FvSstNrNfgQe_QteoTch*u$B8pPYt8#3U6ONuG?YY&gUt$2|yAe"
    "60v9ajl>VD{=Lp2r4DIq#&duX(mNLDJi22CkeV$6-?qql%XU+ud4z{d>oKVkdcd(AXh2G%mgtr)q~{8*v!-"
    "_R4G6Ejy?HKFQ5hSB+R(T!X={4;;R9ZSiE<`h#o_bZWcd8UzK^zzm#RR$VDV_CAlqf8JX&A)-Y9sGXR|eN*Caqmdx^GyTVx>6-"
    "qr$9uQHKpGx8>tsj$$WwZ=kqa>FSRUBV*k!dhpEY1|cTKCWPeC;pei_04t10#D35Cbfuv~$@&2zi3jsSI`!qpv78nf0sjFlX*q#5"
    "I2^z?sqQ$^nE$5as%D?2ryqnb>BP?&Zf{GwLZ%F&6YpC+b?#qfERtEq0kc7WPMsvZ@P`u@SZ)!2SqA0_Y2-"
    "ZfO&2^t_k$#=2}zkpi3`!?Zm?j=q0U*gREOlW?(F2GzU31_s`Rj-m@VV$^ufyCk}_dNHgee?mH-"
    ")l@=zUa!gmaK$}!&!(r{v*~H?Y<k*1o1PBNrKfJ^Y<g;*O;4?}>8X7-Jw2N-%57?lV$(Hbw5M;3y~!thu-ppuPOp!D{y*jtEbj"
)


@lru_cache(maxsize=1)
def _sources() -> dict[str, str]:
    raw = zlib.decompress(base64.b85decode(_PAYLOAD.encode("ascii")))
    if hashlib.sha256(raw).hexdigest() != _PAYLOAD_SHA256:
        raise RuntimeError("Tensor-backprop fixture payload failed its integrity check.")
    sources = json.loads(raw.decode("utf-8"))
    if len(sources) != 155:
        raise RuntimeError("Tensor-backprop fixture payload is incomplete.")
    return sources


def _source_for(number: int) -> tuple[str, str]:
    key = f"{number:03d}"
    try:
        return key, _sources()[key]
    except KeyError as error:
        raise ValueError(f"Unknown tensor-backprop exercise: {number}") from error


def _named_call_statements(source: str, function_name: str) -> list[ast.stmt]:
    tree = ast.parse(source)
    return [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == function_name
    ]


def _run_statements(key: str, statements: list[ast.stmt], namespace: dict, label: str) -> None:
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(
        compile(module, f"<tensor-backprop-{label}-{key}>", "exec"),
        namespace,
        namespace,
    )


def run_fixture(number: int, namespace: dict) -> None:
    """Execute one fully opaque fixture in the notebook's global namespace."""
    key, source = _source_for(number)
    exec(compile(source, f"<tensor-backprop-fixture-{key}>", "exec"), namespace, namespace)


def run_forward_reference(number: int, namespace: dict) -> None:
    """Register a private forward result while using notebook-visible inputs."""
    if not 1 <= number <= 15:
        raise ValueError("Visible forward references exist only for Exercises 001-015.")
    key, source = _source_for(number)
    statements = _named_call_statements(source, "_store_forward")
    if len(statements) != 1:
        raise RuntimeError(f"Exercise {key} does not contain exactly one forward reference.")
    _run_statements(key, statements, namespace, "forward-reference")


def run_gradient_reference(number: int, namespace: dict) -> None:
    """Register private gradient answers while using notebook-visible forward inputs."""
    if not 16 <= number <= 33:
        raise ValueError("Visible local-gradient references exist only for Exercises 016-033.")
    key, source = _source_for(number)
    statements = _named_call_statements(source, "_capture")
    if len(statements) != 1:
        raise RuntimeError(f"Exercise {key} does not contain exactly one gradient reference.")
    _run_statements(key, statements, namespace, "gradient-reference")
