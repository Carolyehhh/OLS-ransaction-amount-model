MacroData = ["""
--Macro Variables
select *
from CMServer.Northwind.dbo.sysallun
where 名稱 in ('消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額年增率(M1B)期底', '貨幣總額(M1B)日平均', '經濟成長率(GDP)–單季',
				'平均每人國內生產毛額(GDP)–美元', '國內生產毛額(GDP)–美元', '躉售物價指數', '躉售物價指數年增率', '貨幣總額年增率(M1B)期底',
				'貨幣總額(M1B)日平均', '貨幣總額(M2)日平均', '中央銀行重貼現率', '台灣失業率(經季節性調整)')
and 年月 >'198701'
order by 年月 asc
""",
"""
-- Market transaction amount
select [日期],[收盤價],[成交金額(千)] as [成交金額(千)_週]
FROM [CMServer].[Northwind].[dbo].sysdbbaseweek
where
[股票代號] = 'TWA00'
AND [日期] >= '19870110'
order by [日期] asc
""",
"""
select [年月],[收盤價],[成交金額(千)] as [成交金額(千)_月]
FROM [CMServer].[Northwind].[dbo].sysdbbasemon
where [股票代號] = 'TWA00'
and 年月 >'198701'
order by [年月] asc
"""
]

# print(type(MacroData)) #str