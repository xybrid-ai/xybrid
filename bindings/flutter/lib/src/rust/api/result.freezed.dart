// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'result.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

/// @nodoc
mixin _$FfiDecisionValue {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiDecisionValueCopyWith<$Res> {
  factory $FfiDecisionValueCopyWith(
          FfiDecisionValue value, $Res Function(FfiDecisionValue) then) =
      _$FfiDecisionValueCopyWithImpl<$Res, FfiDecisionValue>;
}

/// @nodoc
class _$FfiDecisionValueCopyWithImpl<$Res, $Val extends FfiDecisionValue>
    implements $FfiDecisionValueCopyWith<$Res> {
  _$FfiDecisionValueCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiDecisionValue_ChoiceImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_ChoiceImplCopyWith(
          _$FfiDecisionValue_ChoiceImpl value,
          $Res Function(_$FfiDecisionValue_ChoiceImpl) then) =
      __$$FfiDecisionValue_ChoiceImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String key});
}

/// @nodoc
class __$$FfiDecisionValue_ChoiceImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_ChoiceImpl>
    implements _$$FfiDecisionValue_ChoiceImplCopyWith<$Res> {
  __$$FfiDecisionValue_ChoiceImplCopyWithImpl(
      _$FfiDecisionValue_ChoiceImpl _value,
      $Res Function(_$FfiDecisionValue_ChoiceImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? key = null,
  }) {
    return _then(_$FfiDecisionValue_ChoiceImpl(
      key: null == key
          ? _value.key
          : key // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_ChoiceImpl extends FfiDecisionValue_Choice {
  const _$FfiDecisionValue_ChoiceImpl({required this.key}) : super._();

  @override
  final String key;

  @override
  String toString() {
    return 'FfiDecisionValue.choice(key: $key)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_ChoiceImpl &&
            (identical(other.key, key) || other.key == key));
  }

  @override
  int get hashCode => Object.hash(runtimeType, key);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_ChoiceImplCopyWith<_$FfiDecisionValue_ChoiceImpl>
      get copyWith => __$$FfiDecisionValue_ChoiceImplCopyWithImpl<
          _$FfiDecisionValue_ChoiceImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return choice(key);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return choice?.call(key);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(key);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return choice(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return choice?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_Choice extends FfiDecisionValue {
  const factory FfiDecisionValue_Choice({required final String key}) =
      _$FfiDecisionValue_ChoiceImpl;
  const FfiDecisionValue_Choice._() : super._();

  String get key;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_ChoiceImplCopyWith<_$FfiDecisionValue_ChoiceImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionValue_YesNoImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_YesNoImplCopyWith(
          _$FfiDecisionValue_YesNoImpl value,
          $Res Function(_$FfiDecisionValue_YesNoImpl) then) =
      __$$FfiDecisionValue_YesNoImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double pYes});
}

/// @nodoc
class __$$FfiDecisionValue_YesNoImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_YesNoImpl>
    implements _$$FfiDecisionValue_YesNoImplCopyWith<$Res> {
  __$$FfiDecisionValue_YesNoImplCopyWithImpl(
      _$FfiDecisionValue_YesNoImpl _value,
      $Res Function(_$FfiDecisionValue_YesNoImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? pYes = null,
  }) {
    return _then(_$FfiDecisionValue_YesNoImpl(
      pYes: null == pYes
          ? _value.pYes
          : pYes // ignore: cast_nullable_to_non_nullable
              as double,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_YesNoImpl extends FfiDecisionValue_YesNo {
  const _$FfiDecisionValue_YesNoImpl({required this.pYes}) : super._();

  @override
  final double pYes;

  @override
  String toString() {
    return 'FfiDecisionValue.yesNo(pYes: $pYes)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_YesNoImpl &&
            (identical(other.pYes, pYes) || other.pYes == pYes));
  }

  @override
  int get hashCode => Object.hash(runtimeType, pYes);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_YesNoImplCopyWith<_$FfiDecisionValue_YesNoImpl>
      get copyWith => __$$FfiDecisionValue_YesNoImplCopyWithImpl<
          _$FfiDecisionValue_YesNoImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return yesNo(pYes);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return yesNo?.call(pYes);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(pYes);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return yesNo(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return yesNo?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_YesNo extends FfiDecisionValue {
  const factory FfiDecisionValue_YesNo({required final double pYes}) =
      _$FfiDecisionValue_YesNoImpl;
  const FfiDecisionValue_YesNo._() : super._();

  double get pYes;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_YesNoImplCopyWith<_$FfiDecisionValue_YesNoImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionValue_ScoreImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_ScoreImplCopyWith(
          _$FfiDecisionValue_ScoreImpl value,
          $Res Function(_$FfiDecisionValue_ScoreImpl) then) =
      __$$FfiDecisionValue_ScoreImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double value, FfiScoreStatistic statistic});
}

/// @nodoc
class __$$FfiDecisionValue_ScoreImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_ScoreImpl>
    implements _$$FfiDecisionValue_ScoreImplCopyWith<$Res> {
  __$$FfiDecisionValue_ScoreImplCopyWithImpl(
      _$FfiDecisionValue_ScoreImpl _value,
      $Res Function(_$FfiDecisionValue_ScoreImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
    Object? statistic = null,
  }) {
    return _then(_$FfiDecisionValue_ScoreImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as double,
      statistic: null == statistic
          ? _value.statistic
          : statistic // ignore: cast_nullable_to_non_nullable
              as FfiScoreStatistic,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_ScoreImpl extends FfiDecisionValue_Score {
  const _$FfiDecisionValue_ScoreImpl(
      {required this.value, required this.statistic})
      : super._();

  @override
  final double value;
  @override
  final FfiScoreStatistic statistic;

  @override
  String toString() {
    return 'FfiDecisionValue.score(value: $value, statistic: $statistic)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_ScoreImpl &&
            (identical(other.value, value) || other.value == value) &&
            (identical(other.statistic, statistic) ||
                other.statistic == statistic));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value, statistic);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_ScoreImplCopyWith<_$FfiDecisionValue_ScoreImpl>
      get copyWith => __$$FfiDecisionValue_ScoreImplCopyWithImpl<
          _$FfiDecisionValue_ScoreImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return score(value, statistic);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return score?.call(value, statistic);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(value, statistic);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return score(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return score?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_Score extends FfiDecisionValue {
  const factory FfiDecisionValue_Score(
          {required final double value,
          required final FfiScoreStatistic statistic}) =
      _$FfiDecisionValue_ScoreImpl;
  const FfiDecisionValue_Score._() : super._();

  double get value;
  FfiScoreStatistic get statistic;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_ScoreImplCopyWith<_$FfiDecisionValue_ScoreImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiRunResult {
  Object get field0 => throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiResult field0) legacy,
    required TResult Function(FfiDecisionResult field0) decision,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiResult field0)? legacy,
    TResult? Function(FfiDecisionResult field0)? decision,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiResult field0)? legacy,
    TResult Function(FfiDecisionResult field0)? decision,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRunResult_Legacy value) legacy,
    required TResult Function(FfiRunResult_Decision value) decision,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRunResult_Legacy value)? legacy,
    TResult? Function(FfiRunResult_Decision value)? decision,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRunResult_Legacy value)? legacy,
    TResult Function(FfiRunResult_Decision value)? decision,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiRunResultCopyWith<$Res> {
  factory $FfiRunResultCopyWith(
          FfiRunResult value, $Res Function(FfiRunResult) then) =
      _$FfiRunResultCopyWithImpl<$Res, FfiRunResult>;
}

/// @nodoc
class _$FfiRunResultCopyWithImpl<$Res, $Val extends FfiRunResult>
    implements $FfiRunResultCopyWith<$Res> {
  _$FfiRunResultCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiRunResult_LegacyImplCopyWith<$Res> {
  factory _$$FfiRunResult_LegacyImplCopyWith(_$FfiRunResult_LegacyImpl value,
          $Res Function(_$FfiRunResult_LegacyImpl) then) =
      __$$FfiRunResult_LegacyImplCopyWithImpl<$Res>;
  @useResult
  $Res call({FfiResult field0});
}

/// @nodoc
class __$$FfiRunResult_LegacyImplCopyWithImpl<$Res>
    extends _$FfiRunResultCopyWithImpl<$Res, _$FfiRunResult_LegacyImpl>
    implements _$$FfiRunResult_LegacyImplCopyWith<$Res> {
  __$$FfiRunResult_LegacyImplCopyWithImpl(_$FfiRunResult_LegacyImpl _value,
      $Res Function(_$FfiRunResult_LegacyImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiRunResult_LegacyImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as FfiResult,
    ));
  }
}

/// @nodoc

class _$FfiRunResult_LegacyImpl extends FfiRunResult_Legacy {
  const _$FfiRunResult_LegacyImpl(this.field0) : super._();

  @override
  final FfiResult field0;

  @override
  String toString() {
    return 'FfiRunResult.legacy(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiRunResult_LegacyImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiRunResult_LegacyImplCopyWith<_$FfiRunResult_LegacyImpl> get copyWith =>
      __$$FfiRunResult_LegacyImplCopyWithImpl<_$FfiRunResult_LegacyImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiResult field0) legacy,
    required TResult Function(FfiDecisionResult field0) decision,
  }) {
    return legacy(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiResult field0)? legacy,
    TResult? Function(FfiDecisionResult field0)? decision,
  }) {
    return legacy?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiResult field0)? legacy,
    TResult Function(FfiDecisionResult field0)? decision,
    required TResult orElse(),
  }) {
    if (legacy != null) {
      return legacy(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRunResult_Legacy value) legacy,
    required TResult Function(FfiRunResult_Decision value) decision,
  }) {
    return legacy(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRunResult_Legacy value)? legacy,
    TResult? Function(FfiRunResult_Decision value)? decision,
  }) {
    return legacy?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRunResult_Legacy value)? legacy,
    TResult Function(FfiRunResult_Decision value)? decision,
    required TResult orElse(),
  }) {
    if (legacy != null) {
      return legacy(this);
    }
    return orElse();
  }
}

abstract class FfiRunResult_Legacy extends FfiRunResult {
  const factory FfiRunResult_Legacy(final FfiResult field0) =
      _$FfiRunResult_LegacyImpl;
  const FfiRunResult_Legacy._() : super._();

  @override
  FfiResult get field0;

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiRunResult_LegacyImplCopyWith<_$FfiRunResult_LegacyImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiRunResult_DecisionImplCopyWith<$Res> {
  factory _$$FfiRunResult_DecisionImplCopyWith(
          _$FfiRunResult_DecisionImpl value,
          $Res Function(_$FfiRunResult_DecisionImpl) then) =
      __$$FfiRunResult_DecisionImplCopyWithImpl<$Res>;
  @useResult
  $Res call({FfiDecisionResult field0});
}

/// @nodoc
class __$$FfiRunResult_DecisionImplCopyWithImpl<$Res>
    extends _$FfiRunResultCopyWithImpl<$Res, _$FfiRunResult_DecisionImpl>
    implements _$$FfiRunResult_DecisionImplCopyWith<$Res> {
  __$$FfiRunResult_DecisionImplCopyWithImpl(_$FfiRunResult_DecisionImpl _value,
      $Res Function(_$FfiRunResult_DecisionImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiRunResult_DecisionImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as FfiDecisionResult,
    ));
  }
}

/// @nodoc

class _$FfiRunResult_DecisionImpl extends FfiRunResult_Decision {
  const _$FfiRunResult_DecisionImpl(this.field0) : super._();

  @override
  final FfiDecisionResult field0;

  @override
  String toString() {
    return 'FfiRunResult.decision(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiRunResult_DecisionImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiRunResult_DecisionImplCopyWith<_$FfiRunResult_DecisionImpl>
      get copyWith => __$$FfiRunResult_DecisionImplCopyWithImpl<
          _$FfiRunResult_DecisionImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiResult field0) legacy,
    required TResult Function(FfiDecisionResult field0) decision,
  }) {
    return decision(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiResult field0)? legacy,
    TResult? Function(FfiDecisionResult field0)? decision,
  }) {
    return decision?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiResult field0)? legacy,
    TResult Function(FfiDecisionResult field0)? decision,
    required TResult orElse(),
  }) {
    if (decision != null) {
      return decision(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRunResult_Legacy value) legacy,
    required TResult Function(FfiRunResult_Decision value) decision,
  }) {
    return decision(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRunResult_Legacy value)? legacy,
    TResult? Function(FfiRunResult_Decision value)? decision,
  }) {
    return decision?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRunResult_Legacy value)? legacy,
    TResult Function(FfiRunResult_Decision value)? decision,
    required TResult orElse(),
  }) {
    if (decision != null) {
      return decision(this);
    }
    return orElse();
  }
}

abstract class FfiRunResult_Decision extends FfiRunResult {
  const factory FfiRunResult_Decision(final FfiDecisionResult field0) =
      _$FfiRunResult_DecisionImpl;
  const FfiRunResult_Decision._() : super._();

  @override
  FfiDecisionResult get field0;

  /// Create a copy of FfiRunResult
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiRunResult_DecisionImplCopyWith<_$FfiRunResult_DecisionImpl>
      get copyWith => throw _privateConstructorUsedError;
}
