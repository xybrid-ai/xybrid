// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'observability.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

/// @nodoc
mixin _$FfiRedactionRule {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String find, String replacement) replaceLiteral,
    required TResult Function(String key) omitAttribute,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String find, String replacement)? replaceLiteral,
    TResult? Function(String key)? omitAttribute,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String find, String replacement)? replaceLiteral,
    TResult Function(String key)? omitAttribute,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRedactionRule_ReplaceLiteral value)
        replaceLiteral,
    required TResult Function(FfiRedactionRule_OmitAttribute value)
        omitAttribute,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult? Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiRedactionRuleCopyWith<$Res> {
  factory $FfiRedactionRuleCopyWith(
          FfiRedactionRule value, $Res Function(FfiRedactionRule) then) =
      _$FfiRedactionRuleCopyWithImpl<$Res, FfiRedactionRule>;
}

/// @nodoc
class _$FfiRedactionRuleCopyWithImpl<$Res, $Val extends FfiRedactionRule>
    implements $FfiRedactionRuleCopyWith<$Res> {
  _$FfiRedactionRuleCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiRedactionRule_ReplaceLiteralImplCopyWith<$Res> {
  factory _$$FfiRedactionRule_ReplaceLiteralImplCopyWith(
          _$FfiRedactionRule_ReplaceLiteralImpl value,
          $Res Function(_$FfiRedactionRule_ReplaceLiteralImpl) then) =
      __$$FfiRedactionRule_ReplaceLiteralImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String find, String replacement});
}

/// @nodoc
class __$$FfiRedactionRule_ReplaceLiteralImplCopyWithImpl<$Res>
    extends _$FfiRedactionRuleCopyWithImpl<$Res,
        _$FfiRedactionRule_ReplaceLiteralImpl>
    implements _$$FfiRedactionRule_ReplaceLiteralImplCopyWith<$Res> {
  __$$FfiRedactionRule_ReplaceLiteralImplCopyWithImpl(
      _$FfiRedactionRule_ReplaceLiteralImpl _value,
      $Res Function(_$FfiRedactionRule_ReplaceLiteralImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? find = null,
    Object? replacement = null,
  }) {
    return _then(_$FfiRedactionRule_ReplaceLiteralImpl(
      find: null == find
          ? _value.find
          : find // ignore: cast_nullable_to_non_nullable
              as String,
      replacement: null == replacement
          ? _value.replacement
          : replacement // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiRedactionRule_ReplaceLiteralImpl
    extends FfiRedactionRule_ReplaceLiteral {
  const _$FfiRedactionRule_ReplaceLiteralImpl(
      {required this.find, required this.replacement})
      : super._();

  @override
  final String find;
  @override
  final String replacement;

  @override
  String toString() {
    return 'FfiRedactionRule.replaceLiteral(find: $find, replacement: $replacement)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiRedactionRule_ReplaceLiteralImpl &&
            (identical(other.find, find) || other.find == find) &&
            (identical(other.replacement, replacement) ||
                other.replacement == replacement));
  }

  @override
  int get hashCode => Object.hash(runtimeType, find, replacement);

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiRedactionRule_ReplaceLiteralImplCopyWith<
          _$FfiRedactionRule_ReplaceLiteralImpl>
      get copyWith => __$$FfiRedactionRule_ReplaceLiteralImplCopyWithImpl<
          _$FfiRedactionRule_ReplaceLiteralImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String find, String replacement) replaceLiteral,
    required TResult Function(String key) omitAttribute,
  }) {
    return replaceLiteral(find, replacement);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String find, String replacement)? replaceLiteral,
    TResult? Function(String key)? omitAttribute,
  }) {
    return replaceLiteral?.call(find, replacement);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String find, String replacement)? replaceLiteral,
    TResult Function(String key)? omitAttribute,
    required TResult orElse(),
  }) {
    if (replaceLiteral != null) {
      return replaceLiteral(find, replacement);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRedactionRule_ReplaceLiteral value)
        replaceLiteral,
    required TResult Function(FfiRedactionRule_OmitAttribute value)
        omitAttribute,
  }) {
    return replaceLiteral(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult? Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
  }) {
    return replaceLiteral?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
    required TResult orElse(),
  }) {
    if (replaceLiteral != null) {
      return replaceLiteral(this);
    }
    return orElse();
  }
}

abstract class FfiRedactionRule_ReplaceLiteral extends FfiRedactionRule {
  const factory FfiRedactionRule_ReplaceLiteral(
          {required final String find, required final String replacement}) =
      _$FfiRedactionRule_ReplaceLiteralImpl;
  const FfiRedactionRule_ReplaceLiteral._() : super._();

  String get find;
  String get replacement;

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiRedactionRule_ReplaceLiteralImplCopyWith<
          _$FfiRedactionRule_ReplaceLiteralImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiRedactionRule_OmitAttributeImplCopyWith<$Res> {
  factory _$$FfiRedactionRule_OmitAttributeImplCopyWith(
          _$FfiRedactionRule_OmitAttributeImpl value,
          $Res Function(_$FfiRedactionRule_OmitAttributeImpl) then) =
      __$$FfiRedactionRule_OmitAttributeImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String key});
}

/// @nodoc
class __$$FfiRedactionRule_OmitAttributeImplCopyWithImpl<$Res>
    extends _$FfiRedactionRuleCopyWithImpl<$Res,
        _$FfiRedactionRule_OmitAttributeImpl>
    implements _$$FfiRedactionRule_OmitAttributeImplCopyWith<$Res> {
  __$$FfiRedactionRule_OmitAttributeImplCopyWithImpl(
      _$FfiRedactionRule_OmitAttributeImpl _value,
      $Res Function(_$FfiRedactionRule_OmitAttributeImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? key = null,
  }) {
    return _then(_$FfiRedactionRule_OmitAttributeImpl(
      key: null == key
          ? _value.key
          : key // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiRedactionRule_OmitAttributeImpl
    extends FfiRedactionRule_OmitAttribute {
  const _$FfiRedactionRule_OmitAttributeImpl({required this.key}) : super._();

  @override
  final String key;

  @override
  String toString() {
    return 'FfiRedactionRule.omitAttribute(key: $key)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiRedactionRule_OmitAttributeImpl &&
            (identical(other.key, key) || other.key == key));
  }

  @override
  int get hashCode => Object.hash(runtimeType, key);

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiRedactionRule_OmitAttributeImplCopyWith<
          _$FfiRedactionRule_OmitAttributeImpl>
      get copyWith => __$$FfiRedactionRule_OmitAttributeImplCopyWithImpl<
          _$FfiRedactionRule_OmitAttributeImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String find, String replacement) replaceLiteral,
    required TResult Function(String key) omitAttribute,
  }) {
    return omitAttribute(key);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String find, String replacement)? replaceLiteral,
    TResult? Function(String key)? omitAttribute,
  }) {
    return omitAttribute?.call(key);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String find, String replacement)? replaceLiteral,
    TResult Function(String key)? omitAttribute,
    required TResult orElse(),
  }) {
    if (omitAttribute != null) {
      return omitAttribute(key);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiRedactionRule_ReplaceLiteral value)
        replaceLiteral,
    required TResult Function(FfiRedactionRule_OmitAttribute value)
        omitAttribute,
  }) {
    return omitAttribute(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult? Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
  }) {
    return omitAttribute?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiRedactionRule_ReplaceLiteral value)? replaceLiteral,
    TResult Function(FfiRedactionRule_OmitAttribute value)? omitAttribute,
    required TResult orElse(),
  }) {
    if (omitAttribute != null) {
      return omitAttribute(this);
    }
    return orElse();
  }
}

abstract class FfiRedactionRule_OmitAttribute extends FfiRedactionRule {
  const factory FfiRedactionRule_OmitAttribute({required final String key}) =
      _$FfiRedactionRule_OmitAttributeImpl;
  const FfiRedactionRule_OmitAttribute._() : super._();

  String get key;

  /// Create a copy of FfiRedactionRule
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiRedactionRule_OmitAttributeImplCopyWith<
          _$FfiRedactionRule_OmitAttributeImpl>
      get copyWith => throw _privateConstructorUsedError;
}
