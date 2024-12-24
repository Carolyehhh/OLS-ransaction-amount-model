MacroData = ["""
--Macro Variables
select *
from CMServer.Northwind.dbo.sysallun
where 名稱 in ('消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額年增率(M1B)期底', '貨幣總額(M1B)日平均', '經濟成長率(GDP)–單季',
				'平均每人國內生產毛額(GDP)–美元', '國內生產毛額(GDP)–美元', '躉售物價指數', '躉售物價指數年增率', '貨幣總額年增率(M1B)期底',
				'貨幣總額(M1B)日平均', '貨幣總額(M2)日平均', '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率')
and 年月 >'198701'
order by 年月 asc
""",
"""
-- Market transaction amount(week)
select [日期],[收盤價],[成交金額(千)] as [成交金額(千)_週]
FROM [CMServer].[Northwind].[dbo].sysdbbaseweek
where
[股票代號] = 'TWA00'
AND [日期] >= '19870110'
order by [日期] asc
""",
"""
-- Market transaction amount(month)
select [年月],[收盤價],[成交金額(千)] as [成交金額(千)_月]
FROM [CMServer].[Northwind].[dbo].sysdbbasemon
where [股票代號] = 'TWA00'
and 年月 >'198701'
order by [年月] asc
""",
"""
--過年的月份 (從2017開始)
select distinct *
from(
select left(日期, 6) as 年月, 假日名稱, 是否放假
from CMServer.Northwind.dbo.sysfinholiday
where 名稱 like '%金融業休假日%' and 假日名稱='春節'
) a
""" ,
"""
 --交易天數(月)
select left(日期,6) as 年月, count(current_交易天數)  as 交易天數_月
from (
	SELECT 
		日期,
		'transaction days' AS name, 
		COUNT(DISTINCT 日期) AS current_交易天數
	FROM 
		CMServer.Northwind.dbo.sysdbase
	WHERE 
		日期 >= '20000101'
	GROUP BY 
		日期
) a
group by left(日期,6)
order by left(日期,6)
"""
]

# print((MacroData[-1])) #str

