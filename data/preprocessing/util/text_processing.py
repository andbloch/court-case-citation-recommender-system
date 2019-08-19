import os
import spacy
import re
from collections import namedtuple
import pickle

Regex = namedtuple("Regex", ["name", "regexp"])

"""
The regexes used to extract citations are taken from

Url: https://gist.github.com/mlissner/dda7f6677b98b98f54522e271d486781
Original Author: Mike Lissner
Original License: MPL, GPL, LGPL
"""
citation_filters = [
    Regex("U.S. and State Constitutions", "(U\.? ?S\.?) Const\.?,? ?(((a|A)rt\.?|(a|A)mend\.?|(p|P)mbl\.?|(p|P)reamble)( ?[XVI]+))?((, (ȼs|S|&sect;|&#167) ([0-9]+)) ?(, cl\. ([0-9]+)\.?)?)?"),
    Regex("U.S. Supreme Court", "([0-9]|[1-9][0-9]|[1-5][0-9][0-9]) (u\.? ?s\.?) ([0-9]+)(, ([0-9]+))?( \(([0-9]+)\))?"),
    Regex("U.S. Code", "([0-9]+) U\.? ?S\.? ?C\.?( ?S\.?|A\.?)? (ȼs|&sect;|&#167|section|sect?\.?)? ?(\d{1,6}(?:[a-zA-Z]{0,4}(?:\-\d{0,3}[a-zA-Z]?)?)?) ?((?:\([0-9a-zA-Z]\))+)? ?(?:\((\d{4})\))?"),
    Regex("I.R.C. Internal Revenue Code", "I\.? ?R\.? ?C\.? (?:ȼs|&sect;|&#167|section|sect?\.?)? ?(\d{1,6}(?:[a-zA-Z]{0,4}(?:\-\d{0,3}[a-zA-Z]?)?)?) ?((?:\([0-9a-zA-Z]\))+)? ?(?:\((\d{4})\))?"),
    Regex("U.S. Public Laws", "Pub(\.?|lic) ?L(\.?|aw) ?(No\.?)? ?(10[4-9]|11[0-9])-([0-9]+)"),
    Regex("U.S. Statutes at Large", "(1(?:17|18|19|20|21))\ Stat\.\ ([0-9]+)"),
    Regex("Code of Federal Regulations Section", "([0-9]+) (C\.?F\.?R\.?) (ȼs)? ?([0-9a-zA-Z\-]+)\.([0-9a-zA-Z\-]+) ?((?:\([0-9a-zA-Z]\))+)? ?(?:\((\d{4})\))?"),
    Regex("Code of Federal Regulations Part", "([0-9]+) (C\.?F\.?R\.?) (Parts?) ?([0-9a-zA-Z\-]+) ?((\([a-zA-Z0-9]\) ?(\([a-zA-Z0-9]\))?) ?((-[0-9](\([a-zA-Z0-9]\) ?(\([a-zA-Z0-9]\))?)) ?)?(\([IXVixv]+\))?)?((Subpart) ?[a-zA-Z])?(\((\d{4})\))?"),
    Regex("Treasury Regulations", "Treas\.? ?Reg\.? ?(ȼs|&sect;|&#167|section|sect?\.?|Parts?)* ?([0-9aA]+)"),
    Regex("Fed. R. Civ. P.", "(F\.?R\.?C\.?P\.?|Fed\.? ?R(\.?|ule) ?Civ\.? ?Pr?o?c?\.?|Federal Rules? of Civil Procedure) ?(Rule)? ?([0-9]+)?"),
    Regex("Federal Register", "((6|7)[0-9]) (F\.?R\.?|Fed\. ?Reg\.) ([0-9,]+)"),
    Regex("Federal Reporter, Second Series", "(17[8-9]|1[8-9][0-9]|[2-9][0-9][0-9]) ?F\.? ?2d\.? ?([0-9]+)"),
    Regex("Federal Reporter, Third Series", "([1-9]|[1-9][0-9]|[1-9][0-9][0-9]) F\.? ?3d\.? ([0-9]+)"),
    Regex("Federal Reporter, First Series", "([1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-7][0-9]|28[0-1]) F\.? ([0-9]+)(\.|;|,|-|\s)"),
    Regex("Federal Supplement", "([0-9]+ (F\.? ?Supp\.? ?2d\.?|F\.? ?Supp\.?) [0-9]+)"),
    Regex("Fed. R. Evid.", "(FRE | fre |Fed\.? ?R(\.?|ule) ?Evid\.?|Federal Rules? of Evidence)( ?[0-9]+)?"),
    Regex("Fed. R. Crim. P.", "(FRCrP|Fed\.? ?R(\.?|ule) ?Crim\.? ?Pr?o?c?\.?|Federal Rules? of Criminal Procedure) (([0-9]+)\.?([0-9])?)?"),
    Regex("Fed. R. App. P.", "(FRAP|Fed\.? ?R(\.?|ule) ?App\.? ?Pr?o?c?\.?|Federal Rules? of Appellate Procedure) ([0-9]+(\.1)?)?"),
    Regex("Uniform Commercial Code", "(UCC|U\.C\.C\.|Uniform Commercial Code) ?(ȼs|&sect;|&#167;|section|sect?\.?)* ?(([1-9A]+)-([0-9]+))"),
    Regex("Regional State Reporters", "([1-9]|[1-9][0-9]|[1-9][0-9][0-9]) ((So\.?|P\.?|S\.? ?W\.?|S\.? ?E\.?|N\.? ?W\.?|N\.? ?E\.?|A\.?)( ?(2|3)d\.?)?) ([0-9]+)(,|\.|;| )"),
    Regex("Regional State Reporters (U.S. - 3d ser.)", "([1-9][0-9]{0,2}) ((So\.?|P\.?|S\.?W\.?|S\.?E\.?|N\.?W\.?|N\.?E\.?|A\.?) ?3d\.?) ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - P)", "([1-9][0-9]{0,2}) P\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - P2d)", "([1-9][0-9]{0,2}) P\.? ?2d\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - SW)", "([1-9][0-9]{0,2}) S\.? ?W\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - SW2d)", "([1-9][0-9]{0,2}) S\.? ?W\.? ?2d ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - NW)", "([1-9][0-9]{0,2}) N\.? ?W\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - NW2d)", "([1-9][0-9]{0,2}) N\.?\ ?W\.?\ ?2d ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - So.)", "([1-9][0-9]{0,2}) So\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - So2d)", "([1-9][0-9]{0,2}) So\.? ?2d\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - SE)", "([1-9][0-9]{0,2}) S\.? ?E\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - SE2d)", "([1-9][0-9]{0,2}) S\.? ?E\.? ?2d ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - NE)", "([1-9][0-9]{0,2}) N\.? ?E\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - NE2d)", "([1-9][0-9]{0,2}) N\.? ?E\.? ?2d ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - A)", "([1-9][0-9]{0,2}) A\.? ([0-9]+)"),
    Regex("Regional State Reporters (U.S. - A2d)", "([1-9][0-9]{0,2}) A\.? ?2d ([0-9]+)"),
    Regex("Code of Alabama", "(Alabama|Ala\.?) ?Code (ȼs|&sect;|&#167|section|sect?\.?)* ?(([0-9A]+)-([0-9a-z]+)-([-0-9a-z]+))"),
    Regex("Alabama Appellate Courts", "[0-9]+ Ala\.?( ?Civ\.? ?App\.? ?)? [0-9]+"),
    Regex("Alaska Statutes", "Alaska Stat(\.?|utes?) (ȼs|&sect;|&#167|section|sect?\.?)* ?(([0-9]+).([0-9a-z]+).([-0-9a-z]+))"),
    Regex("Alaska Appellate Courts", "([0-9]+) P\.?(2|3)d ([0-9]+)(, ?[-0-9 n\.]+)? \(Alaska ((Ct\.)? ?App\.?)? ?[0-9]+\)"),
    Regex("Arizona Statutes", "Ariz(\.?|ona) (Rev\.?)? ?Stat(\.?|utes?)( Ann\.?)? (ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([-0-9A\.]+)"),
    Regex("Arizona Cases", "[0-9]+ (Ariz\.? ?App\.?|Ariz\.?) [0-9]+"),
    Regex("Arkansas Code", "Ark(\.?|ansas) Code( Ann\.?)? (ȼs|&sect;|&#167|section|sect?\.?)* ?(([0-9A]+)-([0-9a-z]+)-([-0-9a-z]+))"),
    Regex("Arkansas cases", "[0-9]+ (Ark\.? ?App\.?|Ark\.?) [0-9]+"),
    Regex("California Code", "Cal(\.?|ifornia) (agric\.?|bus\.? ?& ?prof\.?|bpc\.?|civ\.? ?proc\.?|ccp\.?|civil|civ\.?|com\.?|corp\.?|edu?c\.?|elec\.?|evid\.?|fam\.?|fin\.?|fish ?& ?game|fgc\.?|food ?& ?agric\.?|fac\.?|govt?\.?|harb\.? ?& ?nav\.?|hnc\.?|health ?& ?safety|hsc\.?|ins\.?|labor|lab\.?|mil\.? ?& ?vet\.?|mvc\.?|penal|pen\.?|prob\.?|pub\.? ?cont\.?|pcc\.?|pub\.? ?res\.?|prc\.?|pub\.? ?util\.?|rev\.? ?& ?tax\.?|rtc\.?|sts\.? ?& ?high\.?|shc\.?|unemp\.? ?ins\.?|uic\.?|veh\.?|water|wat\.?|welf\.? ?& ?inst\.?|wic\.?) Code( Ann\.?)? (ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.?]+)"),
    Regex("California Cases", "([0-9]+ (Cal\.? ?4th|Cal\.? ?3d|Cal\.? ?2d|Cal\.? ?Rptr\.? ?3d|Cal\.? ?Rptr\.? ?2d|Cal\.? ?Rptr\.?|Cal\.? ?App\.? ?4th|Cal\.? ?App\.? ?3d|Cal\.? ?App\.? ?2d) [0-9]+)(,|\.|;| )"),
    Regex("Colorado Statutes", "Colo(\.?|rado) (Rev\.?)? ?Stat(\.?|utes?)( Ann\.?)? (ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9\.]+)"),
    Regex("Connecticut Statutes", "Conn(\.?|ecticut) ?Gen(\.?|eral) ?Stat(\.?|utes?)( Ann\.?)? (ȼs|sect?\.?|&sect;|&#167|section)* ?([0-9]+a?)-([0-9a-z]+)"),
    Regex("Connecticut Cases", "[0-9]+ (Conn\.? ?App\.?|Conn\.?) [0-9]+"),
    Regex("Delaware Code", "Del(\.?|aware) ?Code ?( Ann\.?)?,? tit\.? ([0-9]+)(,? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9]+))?"),
    Regex("Delaware cases", "Del\.? (Ch\.?)? 2[0-9]+"),
    Regex("D.C. Code", "(D\.?C\.?|District of Columbia) Code (Ann.?)?"),
    Regex("D.C. Court of Appeals", "D\.? ?C\.? ?App\.? (1999|2[0-9]+)"),
    Regex("Florida Statutes", "(F\.? ?S\.? ?A\.?|Fla\.? Stat\.?( Ann\.?)?) ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)\.([0-9]+)"),
    Regex("Florida cases", "Fla\.? (Dist\.?)? ?(App\.?)? ?[1-5]? ?(Dist\.)? ?(199|200)[0-9]"),
    Regex("Georgia Code", "Ga\.? Code( Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([-0-9A\.]+)"),
    Regex("Georgia Cases", "([0-9]+ Ga\.?( ?App\.?)? [0-9]+)"),
    Regex("Hawai'i Statutes", "Haw(\.?|ai'?i) Rev(\.?|ised) Stat(\.?|utes)?( Ann\.?)?"),
    Regex("Hawai'i Appellate Courts", "(8[7-9]|9[0-9]|1[0-9]+) Haw(\.?|ai'?i) [0-9]+"),
    Regex("Idaho Code", "Idaho Code ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([0-9]+)"),
    Regex("Idaho Supreme Court", "[0-9]+ Idaho [0-9]+"),
    Regex("Illinois Statutes", "([0-9]+) (ILCS|Ill\.? Comp\.? Stat\.?( Ann\.)?) ([0-9]+)\/([-0-9a-z&#;\.]+)"),
    Regex("Illinois Cases", "([0-9]+ (Ill\.? ?2d|Ill\.? ?Dec\.?|Ill\.? ?App\.? ?2d|Ill\.? ?App\.? ?3d|Ill\.? ?2d\.?) [0-9]+)(,|\.|;| )"),
    Regex("Indiana Code", "(I\.?C\.?|Ind(\.?|iana) ?Code ?(Ann(\.?|otated))?) ?(ȼs|&sect;|&#167|section|sect?\.?)* ?(([0-9\.]+)-([0-9\.]+)-([0-9\.]+)-([0-9\.]+))"),
    Regex("Indiana Cases", "[0-9]+ (Ind\.? ?App\.?|Ind\.?) [0-9]+"),
    Regex("Iowa Cases", "\(Iowa (App\.?)? ?(199(8|9)|200[0-9])\)"),
    Regex("Kansas Statutes", "(K\.?S\.?A\.?|Kan(\.?|sas) ?Stat\.? ?( Ann\.?)?) ?(ȼs|&sect;|&#167|section|sect?\.?)? ?([0-9a]+)-([-0-9a-z,]+)"),
    Regex("Kansas Cases", "[0-9]+ (Kan\.? ?App\.? ?2d\.?|Kan\.?) [0-9]+"),
    Regex("Kentucky Statutes", "K(y\.|entucky) ?Rev(\.|ised) ?Stat(\.|utes) ?( Ann(\.|otated))? ?(ȼs|&sect;|&#167|sect?\.?)* ?([0-9]+)([A-Z])?\.(([0-9]+)-)?([0-9a]+)"),
    Regex("Kentucky Cases", "Ky\.? (App\.?)? ?(199(6|7|8|9)|200[0-9])"),
    Regex("Maine Statutes", "Me\.? ?Rev\.? ?Stat\.? ?(Ann\.?)?,? ?tit(\.?|le) ?([-0-9a-z]+),? (ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9a-z]+)"),
    Regex("Maine Supreme Court", "(199[7-9]|200[0-9]) ME ([0-9]+)"),
    Regex("Maryland Court of Appeals", "(33[7-9]|3[4-9][0-9]|4[0-9][0-9]) Md\.? ?([0-9]+)"),
    Regex("Maryland Court of Special Appeals", "(10[4-9]|1[1-9][0-9]) Md\.? ?App\.? ?([0-9]+)"),
    Regex("Maryland Cases", "[0-9]+ (M\.?D\.? ?App\.?|M\.?D\.?) [0-9]+"),
    Regex("Massachusetts General Laws", "Mass\.? ?Gen\.? ?Laws ?ch\.? ?([0-9A-Z]+),? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)"),
    Regex("Massachusetts SJC Cases", "([2-4][0-9][0-9]) Mass\.? ([0-9]+)"),
    Regex("Massachusetts Ct. App. Cases", "([1-9]|[1-9][0-9]) Mass\.? App\.? Ct\.? ([0-9]+)"),
    Regex("Michigan Compiled Laws", "Mich\.? ?Comp\.? ?Laws ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.]+)"),
    Regex("Michigan Supreme Court", "(([0-9]+) Mich\.?( ?App\.?)? [0-9]+)"),
    Regex("Michigan Court of Appeals", "([0-9]+) Mich\.? ?(Ct\.?)? ?App\.? ?[0-9]+"),
    Regex("Minnesota Statutes", "Minn\.? ?Stat\.? ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9][0-9A-Z\.]+)"),
    Regex("Minnesota Cases", "[0-9]+ (Minn\.?) [0-9]+"),
    Regex("Mississippi Code", "Miss\.? ?Code ?Ann\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([0-9]+)-([0-9]+)"),
    Regex("Mississippi Cases", "Miss\.? ?(Ct\.? ?App\.?)? ?(199[6-9]|200[0-9])"),
    Regex("Montana Code", "Mont\.? ?Code ?Ann\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([0-9]+)-([0-9]+)"),
    Regex("Nebraska Statutes", "Neb(\.?|raska) ?Rev(\.?|ised) ?Stat(\.?|utes) ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)-([0-9\.]+)"),
    Regex("Nevada Statutes", "(N\.? ?R\.? ?S\.?|Nev\.? ?Rev\.? ?Stat\.? ?(Ann\.?)?) ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9A-Z]+)\.([0-9\.]+)"),
    Regex("Nevada cases", "[0-9]+ Nev\.? [0-9]+"),
    Regex("New Jersey Statutes", "(N\.?J\.?S\.?A\.?|N\.?J\.? Stat\.? Ann\.?) ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9a-b]+):([0-9a-b]+)-([0-9a-b\.]+)"),
    Regex("New Jersey Administrative Code", "(N\.?J\.?A\.?C\.?|N\.?J\.? Administrative Code)"),
    Regex("New Jersey Appellate Cases", "[0-9]+ N\.? ?J\.?( ?Super\.?)? [0-9]+"),
    Regex("New Mexico Statutes", "(N\.?M\.?S\.?A\.?|N\.?M\.? ?Stat\.? ?(Ann\.)?) ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9a-z]+)-([0-9a-z]+)-([0-9a-z]+)"),
    Regex("New Mexico Cases", "(199[8-9]|200[0-9]) ?-?(NMCA|NMSC) ?-?([0-9]+)"),
    Regex("New Mexico Reports", "([0-9]+ (N\.? ?M\.?) [0-9]+)"),
    Regex("N.Y. C.P.L.R.", "(New York|N\.?Y\.? ?)?(C\.?P\.?L\.?R\.?|Civil Practice Law and Rules)"),
    Regex("New York Court of Appeals", "(79|[8-9][0-9]|[1-4][0-9][0-9]) N\.?Y\.?2d\.? ?([0-9]+)"),
    Regex("New York Cases", "([0-9]+ (N\.? ?Y\.? ?2d|N\.? ?Y\.? ?S\.? ?2d|A\.? ?D\.? ?(2|3)d) [0-9]+)"),
    Regex("North Carolina Statutes", "N\.? ?C\.? ?Gen\.? ?Stat\.?( ?Ann\.)? (ȼs|S|&sect;|&#167|section|sect?\.?)* ?([0-9a-z]+)-([0-9\.a-z]+)"),
    Regex("North Carolina cases", "([0-9]+ (N\.? ?C\.? ?App\.?|N\.? ?C\.?) [0-9]+)"),
    Regex("North Dakota Code", "N\.?D\.? ?Cent\.? ?Code (ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.]+)-([0-9\.]+)"),
    Regex("North Dakota Supreme Court", "(199[7-9]|200[0-9]) ?ND ?([0-9]+)"),
    Regex("North Dakota Ct. of Appeals", "(199[8-9]|200[0-9]) ?ND App\.? ?([0-9]+)"),
    Regex("North Dakota Cases", "([0-9]+) N\.?W\.?2d\.? ([0-9]+)(, [0-9]+)? ?\(N\.D\. ?(Ct. ?App.)? ?[0-9]+\)"),
    Regex("Ohio Code", "Ohio ?Rev\.? ?Code ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9\.A]+)"),
    Regex("Ohio Administrative Code", "Ohio ?Admin\.? ?Code ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9\.:]+)"),
    Regex("Ohio Supreme Court", "(199[2-9]|20[0-9][0-9])-Ohio-([0-9]+)"),
    Regex("Ohio Cases", "([0-9]+ (Ohio ?St\.? ?3d|Ohio ?St\.? ?2d|Ohio ?St\.?|Ohio App\.? ?3d|Ohio App\.? ?2d|Ohio App\.?|Ohio) [0-9]+)"),
    Regex("Oklahoma Cases", "(19[0-9][0-9]|20[0-9][0-9]) OK [0-9]+"),
    Regex("Oregon Statutes", "Ore?(\.?|egon) ?Rev\.? ?Stat\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9a-d]+)(.([0-9a-d]+))?"),
    Regex("Oregon Cases", "[0-9]+ (Or\.? ?App\.?|Or\.?) [0-9]+"),
    Regex("Pennsylvania Statutes", "([0-9]+) Pa\.?( ?C(ons)?\.?)? ?S(tat)?\.?( ?Ann\.)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)([0-9]{2})(\.([0-9]+))?"),
    Regex("Pennsylvania Code of Regulations", "([0-9]+) Pa\.? ?Code ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)(\.([0-9]+))?"),
    Regex("Pennsylvania Supreme Court", "\(Pa\.? ?(199[7-9]|200[0-9])\)"),
    Regex("Pennsylvania Cases", "(([1-9][0-9][0-9]|[1-9][0-9]|[1-9]) (Pa\.? ?(Super\.?|Superior)( ?Ct\.?)?|Pa\.? ?(Commw\.?|Commonwealth)( ?Ct\.?)?|Pa\.?) [0-9]+)"),
    Regex("Laws of Puerto Rico", "P\.?R\.? ?Laws ?Ann\.?"),
    Regex("General Laws of Rhode Island", "R\.?I\.? ?Gen\.? ?Laws ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.A]+)-([0-9\.]+)-([0-9\.]+)"),
    Regex("Code of Rhode Island Rules", "R\.?I\.? ?Code ?R\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.A]+)-([-0-9\.]+)"),
    Regex("South Carolina Codes", "S\.? ?C\.? ?Code (Ann\.?)? ?(Regs\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.A]+)-([0-9\.]+)(-([0-9\.]+))?"),
    Regex("South Carolina cases", "[0-9]+ (S\.? ?C\.?) [0-9]+"),
    Regex("South Dakota Codified Laws", "S\.? ?D\.? ?Codified ?Laws ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9a-z\.]+)?"),
    Regex("Tennessee Code", "Tenn(\.?|essee) ?Code ?(Ann(\.?|otated))?"),
    Regex("Utah Code", "Utah Code ?Ann\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)([a-z])?-([0-9]+)([a-z])?-([0-9]+)([a-z]+)?"),
    Regex("Utah Cases", "(19|20)[0-9][0-9] UT [0-9]+"),
    Regex("Vermont Statutes", "Vt\.? ?Stat\.? ?Ann\.?,? ?tit\.? ?([0-9A]+), ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9]+)([a-z]+)?"),
    Regex("Vermont Supreme Court", "(15[4-9]|16[0-9]|17[0-8]) Vt\. [0-9]+"),
    Regex("Vermont Code", "Vt\.? ?Stat\.? ?Ann\.? ?tit\.? ?([0-9A]+), ?(ȼs|S|&sect;|&#167|(s|S)ection)* ?([0-9]+)([a-z]+)?"),
    Regex("Virginia Code", "Va\.? ?Code ?Ann\.? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([-0-9A\.:]+)"),
    Regex("Virginia Cases", "[0-9]+ Va\.? ?(App\.?)? ?[0-9]+"),
    Regex("Revised Code of Washington", "Wash\.? ?Rev\.? ?Code? ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9A-Z\.]+)"),
    Regex("Washington Cases", "[0-9]+ (Wash\.? ?2d\.?|Wash\.? ?App\.?|Wash\.?) [0-9]+"),
    Regex("Wisconsin Statutes", "Wis\.? ?Stat\.? ?(Ann\.?)? ?(ȼs|&sect;|&#167|section|sect?\.?)* ?([0-9\.]+)"),
    Regex("Wisconsin Cases", "([0-9]+ (Wis\.? ?2d\.?|Wis\.?) [0-9]+)(,|\.|;| )"),
    Regex("Wisconsin Cases (public domain citation)", "([0-9]+ Wi\.?( ?App\.?)? [0-9]+)"),
    Regex("Miscellaneous Cases", "[0-9]+ (Haw\.? ?App\.?|Haw\.?|Hawaii|Idaho|Mont\.?|Neb\.? ?App\.?|Neb\.?|N\.? ?H\.?|W\.? ?Va\.?|B\.? ?R\.?) [0-9]+"),
    Regex("Public Domain Citations", "(19|20)[0-9][0-9] (ME|MT|ND|SD|VT|WY) [0-9]+"),
    Regex("Congressional Resolutions", "(S\.? ?((Con\.?|J\.?)? ?Res\.?)?|H\.? ?R\.? ?((Con\.?|J\.?)? ?Res\.?)?) ?([0-9]+),? ?(10[3-9]|11[0-9])(th|rd) Cong\."),
    Regex("NLRB Cases", "([0-9]+) N\.?L\.?R\.?B\.? ?(No\.?)? ?([0-9]+)"),
    Regex("BIA Cases", "([0-9]+) I\.? ?& ?N\.? ?Dec\.? ?([0-9]+)"),
    Regex("Decisions of the Comptroller General", "([0-9]+) comp\.? ?gen\.? ?([0-9]+)"),
    Regex("U.S. Patents", "U\.? ?S\.? ?Pat(\.?|ent) ?Nos?\.? ([0-9,]+)"),
    # From here on the regexes are self made
    Regex("Circuit Court", "\([a-z0-9]{0,3}\s?cir\.\s?[0-9]{4}\)"),
    Regex("Circuit Court 2", "\([0-9]{1,2}[a-z]{2}\s?cir\.\s?(?:unit\s?[a-z])?\s?[0-9]{4}"),
    Regex("", "[0-9]{0,4}\s?l\.\s?ed\.\s?2d\s?[0-9]{1,4}\s?(?:\([0-9]{4}\))?"),
    Regex("", "[0-9]{1,4}\s?f\.[0-9][a-z]\s?[0-9]{1,4}(?:, [0-9]{1,4})?"),
    Regex("", "[0-9]{1,4}\s?[a-z]\.[0-9][a-z]\s??(?:at\s?[0-9]{1,4})"),
    Regex("", "[0-9]{1,4}\s?[a-z]\.\s?ct\.\s?[0-9]{1,4}\s?(?:\([0-9]{4}\))?"),
    Regex("", "[0-9]{1,4}\s?u.\s?s\.s? at [0-9]{1,4}"),
    Regex("", "[0-9]{1,4}[a-z]\.\s?[a-z]\.\s?at\s?[0-9]{1,4}"),
    Regex("", "[0-9]{1,4}\s?[a-z]\.\s?[a-z]\.\s?[0-9]{1,4}[a-z]?\s?(?:\([0-9]{4}\))?(?:at\s?[0-9]{1,4})?"),
    Regex("", "[0-9]{1,4}\s?[a-z]\.[a-z]\.\s?[0-9]{1,4}"),
    Regex("", "[0-9]{1,4}\s?id\.\s?at\s?[0-9]{1,4}"),
    Regex("", "[0-9]{1,4}\s?l\.\s?ed\.\s?[0-9]{1,4}\s?(?:\([0-9]{4}\))?")

]


def remove_citations(text):
    """Return text with citations replace by empty string"""
    for regex in citation_filters:
        text = re.sub(regex.regexp, "", text)
    return text


def clean_text(text):
    """Return a cleaned text. Cleanining involves:
    1. convert everything to lowercase
    2. replace newlines, tabs by spaces, replace quot. marks by empty char
    3. replace consecutive spaces by one space
    """
    text = text.lower()
    text = text.translate(text.maketrans({"\n": " ", "\t": " ", "\"": "", "'": ""}))
    return re.sub(" +", " ", text)


def remove_judges(text, names):
    """Return a text with all judges in names replaced by empty string"""
    for judge in names:
        text = text.replace(judge, "")
    return text


def get_sentences(text):
    """
    split document into list of sentences
    where a sentence is a list of words
    """
    doc = nlp(text)
    sentences = [str(sent).replace(".", "").split(" ") for sent in doc.sents]
    return sentences


# preprocessing: map these sentences to their corresponding vocabulary index
# encode: text -> index
# count how often unknown occurred
unknown_counter = 0
def word_to_idx(word):
    global unknown_counter
    try:
        return word2idx[word]
    except KeyError:
        # use index of "unknown" placeholder    # =vocabulary[-1]
        unknown_counter += 1
        return word2idx["<unk>"]


# load spacy only with sentencizer (should be a lot faster)
# to split document in stences (list of lists of words)
nlp = spacy.load("en")
nlp.disable_pipes(*[name for name, object in nlp.pipeline])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# read vocabulary an word to index mapping
vocabulary = pickle.load(open(os.path.join(CURR_DIR, '../../pretrained_word_embedding/glove.6B/vocabulary.pkl'), 'rb'))
word2idx = pickle.load(open(os.path.join(CURR_DIR, '../../pretrained_word_embedding/glove.6B/word2idx.pkl'), 'rb'))


def process_text(opinion_text, judges):
    # clean text
    text = clean_text(opinion_text)
    # remove citations
    text = remove_citations(text)
    # remove judges
    text = remove_judges(text, judges)
    # get sentences from text
    text = get_sentences(text)
    # transform list of words into lists of word indices
    text = [list(map(word_to_idx, sentence)) for sentence in text]
    return text
